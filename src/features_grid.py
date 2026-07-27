"""
src/features_grid.py — 격자 간 산포 · 공간 경도 피처
==============================================================================
왜 이 모듈이 있는가 (step23 · §3.20)

  process_weather_data 가 만드는 것은 두 종류뿐이었음.
      {prefix}_g{id}_{var}   개별 격자 (LDAPS 16 / GFS 9)
      {prefix}_mean_{var}    균등평균 1개
  **격자들이 서로 얼마나 다른지, 어느 방향으로 기울어져 있는지를 나타내는 양이 없었음.**

  `_gstd` 가 크다 = "지금 이 시각은 균등평균을 믿으면 안 된다" 는 신호임.
  `_gdx`/`_gdy` 는 공간 경도이며 **능선 가속을 직접 나타내는 양**임.

측정 결과 (step23 · step23b, 그룹 독립 구성 · top-k 200)
      raw OOF  +0.0023   연도별 [+0.0021 +0.0020 +0.0029]  3/3 양수
      표준편차 0.0005 (신뢰비율 4.6)  ·  그룹별 G1 +0.0021 / G2 +0.0013 / **G3 +0.0039**
      => §3.20 의 사전 예측("공간 범위가 1.7배 넓은 G3 에서 최대")이 적중함.

⚠ 선택 예산과 **대체 관계**임 (step23b 상호작용 −0.0027, 3/3, 비율 2.46).
   top-k 를 250 으로 올리면 산포 효과가 −0.0004 로 소멸함.
   K=201~250 에서 복귀하는 개별 격자 컬럼들이 같은 정보를 담고 있기 때문임.
   **따라서 이 피처를 쓸 때 top-k 는 200 을 유지해야 함.**

   보정된 이해: 트리는 격자 산포를 *정확히는* 못 만들지만 개별 격자 컬럼이 충분하면
   여러 분할로 부분 근사함. 이 피처의 가치는 '없는 정보를 준다' 가 아니라
   **'같은 정보를 1개 컬럼으로 압축해 선택 예산을 절약한다'** 임.

주의
  * `build_full_feature_pipeline` **이후에** 호출해야 함 (가지치기를 통과한 격자 컬럼을 씀).
  * train 과 inference 에서 **반드시 동일하게** 호출해야 함. 한쪽만 넣으면
    `feature_cols.pkl` reindex 에서 전부 NaN 이 되어 조용히 성능이 무너짐.
    -> `save_model_dir/grid_features.json` 마커로 자동 동기화함.
==============================================================================
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 산포를 계산할 기저 변수. §3.4 생존 계열(U성분·복사) + 안정도 대리변수.
# **고정 목록이므로 폴드를 참조하지 않음 = 선택 누설 없음.**
BASE = {
    "ldaps": ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
              "etc_0_blh", "heightAboveGround_2_t", "surface_0_NDNLW"],
    "gfs": ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
            "heightAboveGround_100_100u", "heightAboveGround_100_100v",
            "surface_0_gust"],
}
# 격자별로 합성해 추가할 풍속 (u, v, 이름)
WSPAIR = {"ldaps": ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
          "gfs": ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100")}

MARKER = "grid_features.json"


# ------------------------------------------------------------------ 격자 메타
def grid_meta(raw_grid_df):
    """원본 long 포맷에서 grid_id -> (위도, 경도). 공간 경도의 설계행렬용."""
    g = raw_grid_df.drop_duplicates("grid_id").sort_values("grid_id")
    return (g["grid_id"].to_numpy(),
            g["latitude"].to_numpy(float),
            g["longitude"].to_numpy(float))


def build_metas(ldaps_raw, gfs_raw):
    """train.py / inference.py 가 이미 읽어 둔 원본 csv 두 개를 그대로 넘기면 됨."""
    return {"ldaps": grid_meta(ldaps_raw), "gfs": grid_meta(gfs_raw)}


# ------------------------------------------------------------------ 본체
def add_grid_dispersion(df, metas, verbose=True):
    """
    격자 간 산포와 공간 경도를 계산해 DataFrame 으로 반환함 (df 를 수정하지 않음).

    공간 경도는 격자 위경도에 대한 최소제곱임.
        v(grid) ≈ a + b·x + c·y      x = 동서 km, y = 남북 km (중심 기준)
        b = _gdx (동서 경도),  c = _gdy (남북 경도)
    설계행렬이 시간에 대해 고정이므로 유사역행렬을 한 번만 구해 행렬곱 1회로 처리함.
    """
    new, n_var = {}, 0
    for prefix, (gids, lat, lon) in metas.items():
        # 위경도를 km 로 (중심 기준). 이 위도대(37.28°)에서 경도 1° ≈ 88.6 km
        x = (lon - lon.mean()) * 111.32 * np.cos(np.radians(float(lat.mean())))
        y = (lat - lat.mean()) * 110.57
        A = np.vstack([np.ones(len(gids)), x, y]).T
        P = np.linalg.pinv(A)                       # (3, n_grid) — 고정 연산자

        def stack(var):
            cols = [f"{prefix}_g{g}_{var}" for g in gids]
            cols = [c for c in cols if c in df.columns]
            if len(cols) < len(gids) * 0.8:         # 가지치기로 대부분 사라진 변수는 건너뜀
                return None
            return df[cols].to_numpy(float)         # (n_row, n_grid)

        mats = {v: stack(v) for v in BASE[prefix]}
        ua, va, wsname = WSPAIR[prefix]
        if mats.get(ua) is not None and mats.get(va) is not None:
            mats[wsname] = np.sqrt(mats[ua] ** 2 + mats[va] ** 2)

        for v, M in mats.items():
            if M is None:
                continue
            n_var += 1
            mu = np.nanmean(M, axis=1)
            sd = np.nanstd(M, axis=1, ddof=1)
            new[f"{prefix}_{v}_gstd"] = sd                                   # 격자 간 표준편차
            new[f"{prefix}_{v}_grange"] = np.nanmax(M, axis=1) - np.nanmin(M, axis=1)
            co = np.nan_to_num(M, nan=0.0) @ P.T                             # (n_row, 3)
            new[f"{prefix}_{v}_gdx"] = co[:, 1]                              # 동서 경도
            new[f"{prefix}_{v}_gdy"] = co[:, 2]                              # 남북 경도
            if v == wsname:                                                  # 풍속에만
                new[f"{prefix}_{v}_gcv"] = sd / (np.abs(mu) + 1e-6)          # 상대 산포
                new[f"{prefix}_{v}_ggrad"] = np.hypot(co[:, 1], co[:, 2])    # 경도 크기
                new[f"{prefix}_{v}_gstd_x_mean"] = sd * mu                   # 산포 × 수준

    out = pd.DataFrame(new, index=df.index).replace([np.inf, -np.inf], np.nan)
    if verbose:
        print(f"🗺️  격자 산포 피처 {out.shape[1]}개 생성 (기저 {n_var}변수 × "
              f"gstd/grange/gdx/gdy [+풍속은 gcv/ggrad/상호작용])")
    return out


def attach(df, ldaps_raw, gfs_raw, verbose=True):
    """한 줄 헬퍼. train.py / inference.py 양쪽에서 이것만 부르면 됨."""
    metas = build_metas(ldaps_raw, gfs_raw)
    return pd.concat([df, add_grid_dispersion(df, metas, verbose=verbose)], axis=1) \
             .replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ 마커
def save_marker(save_dir, enabled):
    """train 이 켰는지 여부를 저장함. inference 가 이걸 읽어 자동으로 맞춤."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"enabled": bool(enabled)}, open(Path(save_dir) / MARKER, "w"))


def marker_enabled(model_dir):
    p = Path(model_dir) / MARKER
    if not p.exists():
        return False
    try:
        return bool(json.load(open(p)).get("enabled", False))
    except Exception:
        return False