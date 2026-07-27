"""
src/features_obs.py — ASOS 공공 관측 피처 (블록 상수)
==============================================================================
왜 (§8.8 · §3.30)

  역대 LB 상승은 전부 '정보형'이었음. 관측은 NWP 로부터 유도 불가능한 새 정보원이고
  (§3.8 따름정리에 안 걸림), 각 행의 예측기준시점 이전 실측만 쓰므로 2025 의 실제
  체제를 행별로 합법 추적하는 유일한 통로임.

규칙 (규칙 3 실측 조항이 명문 허용)
  "기상 관측자료 등 실측 데이터는 예측기준시점 전에 이미 관측·공개·확정된 과거 자료에
   한해 사용할 수 있습니다."
  * 각 예보 블록의 기준시점 = data_available_kst_dtm (전날 13:00).
  * 이 모듈은 기준시점 **−1시간(12:00)까지**의 관측만 씀 — 공개 지연 안전 여유.
  * 한 블록의 24행은 같은 기준시점을 가지므로 관측 피처는 **블록 상수**임 (합법의 핵심).

피처 10개 (사전 등록 — §3.21: 컬럼 수가 곧 비용)
  창 = 기준시점 직전 24h (D-2 13:00 ~ D-1 12:00). 관측소 평균(수 km 내 인접).
    obs_ws_mean24 / obs_ws_max24 / obs_ws_std24    바람 수준·극값·변동
    obs_ws_last3h                                   최신 상태 (10~12시 평균)
    obs_wd_sin24 / obs_wd_cos24                     풍향 원형평균 (풍속 가중)
    obs_ta_mean24 / obs_hm_mean24                   기온·습도 (드리프트 계열의 실측)
    obs_pa_tend24                                   기압 24h 변화 (종관 변화 신호)
    obs_nwp_bias24                                  ★ 관측 − LDAPS 예보 (D-1 01~12시 평균)
                                                    = 최근 NWP 편차. 진짜 새 정보

추론 경계: 2025-01-01 블록의 편차 계산에는 2024-12-31 의 LDAPS 예보값이 필요함
  -> 학습 때 프레임 꼬리 48h 를 obs_state.csv 로 저장, 추론이 이어붙임 (features_anom 방식).
  관측 csv 는 외부 파일이라 전 기간을 덮으므로 그쪽 경계는 없음.
==============================================================================
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARKER = "obs_features.json"
STATE = "obs_state.csv"
LDAPS_U = "ldaps_mean_heightAboveGround_10_10u"
LDAPS_V = "ldaps_mean_heightAboveGround_10_10v"
MIN_HOURS = 12          # 24h 창에서 관측이 이보다 적으면 NaN


def load_obs(path):
    """asos_hourly.csv -> 시각별 관측소 평균 (ws, wd 원형, ta, hm, pa)."""
    o = pd.read_csv(path, encoding="utf-8-sig")
    o["kst_dtm"] = pd.to_datetime(o["kst_dtm"])
    rad = np.radians(o["wd"].to_numpy(float))
    w = o["ws"].to_numpy(float)
    o["_sin"] = np.sin(rad) * w
    o["_cos"] = np.cos(rad) * w
    g = o.groupby("kst_dtm").agg(ws=("ws", "mean"), ta=("ta", "mean"),
                                 hm=("hm", "mean"), pa=("pa", "mean"),
                                 s=("_sin", "mean"), c=("_cos", "mean"),
                                 n=("ws", "count"))
    return g


def _block_features(avails, obs, frame_ws):
    """
    avails   : 정렬된 고유 기준시점 배열 (datetime64)
    obs      : load_obs 결과 (kst_dtm 인덱스)
    frame_ws : Series (forecast_kst_dtm 인덱스) — LDAPS ws10 예보값 (편차 계산용)
    반환: DataFrame (index=avails, 10 컬럼)
    """
    oidx = obs.index
    rows = {}
    for A in avails:
        A = pd.Timestamp(A)
        w0, w1 = A - pd.Timedelta(hours=24), A - pd.Timedelta(hours=1)
        win = obs.loc[(oidx >= w0) & (oidx <= w1)]
        r = {}
        if len(win) >= MIN_HOURS:
            ws = win["ws"].to_numpy(float)
            r["obs_ws_mean24"] = np.nanmean(ws)
            r["obs_ws_max24"] = np.nanmax(ws)
            r["obs_ws_std24"] = np.nanstd(ws, ddof=1)
            last3 = win.loc[win.index >= A - pd.Timedelta(hours=3), "ws"]
            r["obs_ws_last3h"] = float(np.nanmean(last3)) if len(last3) else np.nan
            s, c = np.nansum(win["s"]), np.nansum(win["c"])
            norm = float(np.hypot(s, c)) + 1e-9
            r["obs_wd_sin24"], r["obs_wd_cos24"] = s / norm, c / norm
            r["obs_ta_mean24"] = np.nanmean(win["ta"])
            r["obs_hm_mean24"] = np.nanmean(win["hm"])
            pa = win["pa"].dropna()
            r["obs_pa_tend24"] = float(pa.iloc[-1] - pa.iloc[0]) if len(pa) >= 2 else np.nan
        else:
            for k in ["obs_ws_mean24", "obs_ws_max24", "obs_ws_std24", "obs_ws_last3h",
                      "obs_wd_sin24", "obs_wd_cos24", "obs_ta_mean24", "obs_hm_mean24",
                      "obs_pa_tend24"]:
                r[k] = np.nan
        # ★ 최근 NWP 편차: D-1 01:00~12:00 (기준시점 당일 새벽~정오) 예보 vs 관측
        f0, f1 = A - pd.Timedelta(hours=12), A - pd.Timedelta(hours=1)
        ft = frame_ws.loc[(frame_ws.index >= f0) & (frame_ws.index <= f1)]
        if len(ft) >= 6:
            ov = obs["ws"].reindex(ft.index)
            d = (ov - ft).dropna()
            r["obs_nwp_bias24"] = float(d.mean()) if len(d) >= 6 else np.nan
        else:
            r["obs_nwp_bias24"] = np.nan
        rows[A] = r
    return pd.DataFrame(rows).T


def attach(df, obs_path, prior_frame=None, verbose=True):
    """
    df : forecast_kst_dtm, data_available_kst_dtm, LDAPS u/v mean 컬럼을 가진 프레임
    prior_frame : 추론 시 학습 꼬리 (obs_state.csv 로드 결과)
    """
    obs = load_obs(obs_path)
    fr = df[["forecast_kst_dtm", LDAPS_U, LDAPS_V]].copy()
    fr["forecast_kst_dtm"] = pd.to_datetime(fr["forecast_kst_dtm"])
    if prior_frame is not None:
        pf = prior_frame.copy()
        pf["forecast_kst_dtm"] = pd.to_datetime(pf["forecast_kst_dtm"])
        fr = pd.concat([pf[fr.columns], fr], ignore_index=True)
    fr = fr.drop_duplicates("forecast_kst_dtm").sort_values("forecast_kst_dtm")
    frame_ws = pd.Series(np.hypot(fr[LDAPS_U].to_numpy(float), fr[LDAPS_V].to_numpy(float)),
                         index=fr["forecast_kst_dtm"].to_numpy())

    av = pd.to_datetime(df["data_available_kst_dtm"])
    blocks = np.sort(av.dropna().unique())
    bf = _block_features(blocks, obs, frame_ws)
    out = df.copy()
    m = av.map(lambda a: bf.index.get_indexer([a])[0] if pd.notna(a) else -1)
    for c in bf.columns:
        v = np.full(len(df), np.nan)
        ok = m.to_numpy() >= 0
        v[ok] = bf[c].to_numpy()[m.to_numpy()[ok]]
        out[c] = v
    if verbose:
        nn = int(out[list(bf.columns)].notna().all(axis=1).sum())
        print(f"📡 관측 피처 {bf.shape[1]}개 (블록 {len(blocks):,}) — 유효 {nn:,}/{len(df):,}행")
    return out


# ------------------------------------------------------------------ 상태/마커
def save_state(save_dir, df):
    """프레임 꼬리 48h (LDAPS 예보값) — 추론의 편차 계산 경계용."""
    fr = df[["forecast_kst_dtm", LDAPS_U, LDAPS_V]].copy()
    fr["forecast_kst_dtm"] = pd.to_datetime(fr["forecast_kst_dtm"])
    fr.sort_values("forecast_kst_dtm").tail(48) \
      .to_csv(Path(save_dir) / STATE, index=False, encoding="utf-8-sig")
    json.dump({"enabled": True}, open(Path(save_dir) / MARKER, "w"))


def attach_test(df_test, model_dir, obs_path, verbose=True):
    prior = pd.read_csv(Path(model_dir) / STATE, encoding="utf-8-sig")
    return attach(df_test, obs_path, prior_frame=prior, verbose=verbose)


def marker_enabled(model_dir):
    p = Path(model_dir) / MARKER
    if not p.exists():
        return False
    try:
        return bool(json.load(open(p)).get("enabled", False))
    except Exception:
        return False
