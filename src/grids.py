"""
src/grids.py — 격자 -> KPX 그룹 가중 추출
==============================================================================
지금 파이프라인은 ldaps_mean_* (16격자 균등평균) 을 쓴다. 그런데 격자별로
그룹까지의 거리가 크게 다르고 표고도 869~1001 m 로 130 m 차이가 난다.
터빈은 전부 능선 정상부(표고 ~1000 m 격자)에 있는데 계곡 격자를 같은 무게로
섞고 있는 셈이다. 그룹을 가르는 몇 안 되는 신호라 반드시 살려야 한다.

실측 좌표로 계산한 상위 격자
   G1: g5(0.26) g6(0.25) g10(0.08) g11(0.08)
   G2: g6(0.28) g11(0.19) g7(0.09)  g12(0.09)
   G3: g12(0.35) g11(0.15) g16(0.07) g7(0.07)
==============================================================================
"""
import numpy as np
import pandas as pd


def _dms(d, m, s):
    return d + m / 60 + s / 3600


TURBINES = {
    "kpx_group_1": [(_dms(37,16,55.61), _dms(128,57,2.10)), (_dms(37,17,4.05), _dms(128,56,58.35)),
                    (_dms(37,17,11.49), _dms(128,56,58.99)), (_dms(37,17,23.11), _dms(128,57,3.68)),
                    (_dms(37,17,28.20), _dms(128,57,15.58)), (_dms(37,17,19.48), _dms(128,57,24.96))],
    "kpx_group_2": [(_dms(37,17,16.20), _dms(128,57,34.67)), (_dms(37,17,11.29), _dms(128,57,47.24)),
                    (_dms(37,17,0.97), _dms(128,57,57.44)), (_dms(37,16,52.77), _dms(128,58,4.18)),
                    (_dms(37,16,44.89), _dms(128,58,1.12)), (_dms(37,16,30.58), _dms(128,58,2.54))],
    "kpx_group_3": [(_dms(37,16,59.73), _dms(128,57,44.97)), (_dms(37,16,40.41), _dms(128,58,13.80)),
                    (_dms(37,16,28.03), _dms(128,58,22.54)), (_dms(37,16,18.58), _dms(128,58,29.01)),
                    (_dms(37,16,6.83), _dms(128,58,35.68))],
}
HUB_BASE_M = 990.0          # 터빈 입지 표고 (능선 정상부)


def _km(la1, lo1, la2, lo2):
    return np.hypot((la1 - la2) * 110.57, (lo1 - lo2) * 111.32 * np.cos(np.radians(37.28)))


def grid_weights(grid_meta, power=2.0, elev_scale=None, min_km=0.3):
    """
    grid_meta : DataFrame[grid_id, latitude, longitude, (surface_0_h)]
                예: ldaps_train.drop_duplicates('grid_id')
    elev_scale: 표고 차이 페널티(m). None 이면 거리만. 200 정도가 무난.
    반환: DataFrame[grid_id, kpx_group_1, kpx_group_2, kpx_group_3]  (열 합 = 1)
    """
    g = grid_meta.drop_duplicates("grid_id").sort_values("grid_id").reset_index(drop=True)
    W = {"grid_id": g["grid_id"].to_numpy()}
    for grp, pts in TURBINES.items():
        d = np.array([np.mean([_km(la, lo, r.latitude, r.longitude) for la, lo in pts])
                      for r in g.itertuples()])
        w = 1.0 / np.maximum(d, min_km) ** power
        if elev_scale and "surface_0_h" in g:
            dz = np.abs(g["surface_0_h"].to_numpy(float) - HUB_BASE_M)
            w = w * np.exp(-(dz / elev_scale) ** 2)          # 능선 격자에 가중
        W[grp] = w / w.sum()
    return pd.DataFrame(W)


def weighted_extract(raw_grid_df, weights, prefix, value_cols=None,
                     time_col="forecast_kst_dtm", avail_col="data_available_kst_dtm"):
    """
    격자 long 포맷 -> 그룹별 가중평균 wide 포맷.
    반환 컬럼: {prefix}_{group}_{var}   예: ldaps_kpx_group_1_heightAboveGround_10_10u
    기존 {prefix}_mean_* 와 함께 쓰면 된다 (균등평균도 나름의 정보라 지우지 말 것).
    """
    df = raw_grid_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    skip = {time_col, avail_col, "grid_id", "latitude", "longitude"}
    value_cols = value_cols or [c for c in df.columns if c not in skip]

    w = weights.set_index("grid_id")
    out = None
    for grp in [c for c in w.columns]:
        wm = df["grid_id"].map(w[grp]).to_numpy()[:, None]
        num = pd.DataFrame(df[value_cols].to_numpy(float) * wm, columns=value_cols)
        num[time_col] = df[time_col].to_numpy()
        agg = num.groupby(time_col).sum()                    # 가중치 합=1 이므로 합=가중평균
        agg.columns = [f"{prefix}_{grp}_{c}" for c in value_cols]
        out = agg if out is None else out.join(agg)
    return out.reset_index()