"""
src/features_anom.py — 후행 이동평균 편차(anomaly) 피처
==============================================================================
왜 (step31 · §3.31)

  연도 간 드리프트가 실재하고 단조임: 습도 +8%p/3년, 서풍 전 고도 −15~24%, 기온 +1.1K.
  트리가 배운 절대 문턱("습도 70% 이상이면 분기")은 분포가 통째로 밀리면 뜻이 달라짐.
  피처 제외(step31 PART C)와 표본 가중(v18)은 둘 다 기각됨 — 남은 대응은 **표현**임.

      anom720_x = x − (직전 720시간 이동평균)

  최근 30일 체제 대비 편차는 분포가 밀려도 뜻이 유지됨 (표류를 피처 자신이 뺌).
  2025 에서는 이동평균이 2025 체제를 따라가므로, 모델이 학습한 편차 문턱이 그대로 유효함.

규칙 (§6)
  * 후행(과거 방향) 참조만 씀 — 이전 예보 블록들은 전부 현재 행의 예측기준시점 이전에
    공개됐음. 기존 lag/rolling 피처와 같은 원리로 합법.
  * 추론 시 2025년 1월의 창은 2024년 12월(학습 기간) 데이터로 채움 — 과거 자료라 합법.
    학습 때 마지막 720시간을 anom_state.csv 로 저장해 두고 추론이 이어붙임
    (test 시작 2025-01-01 01:00 은 train 끝 2025-01-01 00:00 과 연속).

설계 (사전 등록 — §3.21 교훈: 컬럼 수가 곧 비용)
  * 기저 7계열만: 드리프트 확인된 계열의 _mean_ 변수 (습도 2, U성분 2, 기온 1, 풍속 2)
  * 창 720h(30일) 하나, min_periods 240. 격자 탐색 없음.
  * train / inference 동기화는 features_grid 와 같은 마커 방식 (anom_features.json)
==============================================================================
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

WIN, MINP = 720, 240
MARKER = "anom_features.json"
STATE = "anom_state.csv"

# (이름, 컬럼) — 단일 컬럼 기저
BASE = [
    ("ldaps_r",   "ldaps_mean_heightAboveGround_2_r"),
    ("gfs_r",     "gfs_mean_heightAboveGround_2_2r"),
    ("ldaps_u10", "ldaps_mean_heightAboveGround_10_10u"),
    ("gfs_u100",  "gfs_mean_heightAboveGround_100_100u"),
    ("ldaps_t2",  "ldaps_mean_heightAboveGround_2_t"),
]
# (이름, u컬럼, v컬럼) — 합성 풍속 기저
WS = [
    ("ldaps_ws10", "ldaps_mean_heightAboveGround_10_10u", "ldaps_mean_heightAboveGround_10_10v"),
    ("gfs_ws100",  "gfs_mean_heightAboveGround_100_100u", "gfs_mean_heightAboveGround_100_100v"),
]


def _base_series(df):
    """df 에서 기저 7계열을 (이름 -> ndarray, df 행순서) 로 뽑음. 없는 컬럼은 건너뜀."""
    out = {}
    for name, col in BASE:
        if col in df.columns:
            out[name] = df[col].to_numpy(dtype=float)
    for name, uc, vc in WS:
        if uc in df.columns and vc in df.columns:
            out[name] = np.hypot(df[uc].to_numpy(dtype=float), df[vc].to_numpy(dtype=float))
    return out


def attach(df, prior=None, verbose=True):
    """
    후행 720h 이동평균 편차를 붙여 반환함 (df 수정 없음, 새 컬럼 anom720_*).

    prior : DataFrame(forecast_kst_dtm + 기저 컬럼) — 추론 시 학습 꼬리를 이어붙일 때.
            시간이 df 보다 전부 앞서야 함.
    """
    t = pd.to_datetime(df["forecast_kst_dtm"])
    order = np.argsort(t.to_numpy(), kind="stable")           # 시간순 계산 후 원순서 복원
    inv = np.empty_like(order); inv[order] = np.arange(len(order))

    cur = _base_series(df)
    pri = _base_series(prior) if prior is not None else {}
    n_pri = len(prior) if prior is not None else 0

    new = {}
    for name, sv in cur.items():
        v = sv[order]
        if name in pri:
            v = np.concatenate([pri[name], v])
        roll = pd.Series(v).rolling(WIN, min_periods=MINP).mean().to_numpy()
        anom = (v - roll)[n_pri:] if name in pri else (v - roll)
        new[f"anom720_{name}"] = anom[inv]
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    if verbose:
        nn = int(pd.DataFrame(new).notna().all(axis=1).sum())
        print(f"🌊 후행편차 피처 {len(new)}개 (창 {WIN}h) — 유효 {nn:,}/{len(df):,}행"
              + ("" if prior is None else f"  (학습 꼬리 {n_pri}행 이어붙임)"))
    return out


# ------------------------------------------------------------------ 상태 저장/복원
def save_state(save_dir, df):
    """학습 프레임의 마지막 WIN 시간을 저장 — 추론이 2025년 1월 창을 채우는 데 씀."""
    t = pd.to_datetime(df["forecast_kst_dtm"])
    tail = df.iloc[np.argsort(t.to_numpy(), kind="stable")].tail(WIN)
    cols = ["forecast_kst_dtm"] + [c for _, c in BASE if c in df.columns] \
           + [c for _, u, v in WS for c in (u, v) if c in df.columns]
    tail[list(dict.fromkeys(cols))].to_csv(Path(save_dir) / STATE, index=False, encoding="utf-8-sig")
    json.dump({"enabled": True, "win": WIN}, open(Path(save_dir) / MARKER, "w"))


def attach_test(df_test, model_dir, verbose=True):
    """추론용: 저장된 학습 꼬리를 이어붙여 계산."""
    prior = pd.read_csv(Path(model_dir) / STATE, encoding="utf-8-sig")
    prior["forecast_kst_dtm"] = pd.to_datetime(prior["forecast_kst_dtm"])
    return attach(df_test, prior=prior, verbose=verbose)


def marker_enabled(model_dir):
    p = Path(model_dir) / MARKER
    if not p.exists():
        return False
    try:
        return bool(json.load(open(p)).get("enabled", False))
    except Exception:
        return False
