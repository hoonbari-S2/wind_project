"""
src/features_physics.py — 배치 1: 물리 핵심 피처
==============================================================================
바람 → 밀도 → 파워커브로 이어지는 물리 사슬만 담는다.
같은 계층의 다른 모듈:
    src/features.py         v6 도메인 피처 (기존)
    src/features_env.py     배치 2 — 격자 가중 / 안정도 / 풍향 / lead time (예정)
==============================================================================
하나의 인과 사슬을 코드로 옮긴다.

    바람(허브고도) → 밀도 보정 → 파워커브 → 출력

기존 features.py 는 이 사슬이 두 군데 끊겨 있다.
  1) 허브고도 풍속을 GFS(격자 25km) 10m/100m 두 점으로만 뽑는다.
     LDAPS 는 1.5km 인데 50m 바람을 '난류 변동폭' 으로만 쓰고 풍속으로는 안 쓴다.
  2) 정격 평탄화가 없다. v^3 는 정격(10~11 m/s) 아래에서만 맞는데
     그 위에서는 발전량이 상수다. SCADA 로 이미 확인한 사실이 코드에 없다.

기존 파이프라인을 대체하지 않고 '추가' 한다.
    df = build_full_feature_pipeline(df)      # 기존
    df = add_batch1_features(df)              # 이 모듈
==============================================================================
"""
import numpy as np
import pandas as pd

HUB = 117.0
ROTOR_R = {"vestas": 63.0, "unison": 68.0}      # V126 / U136
TURB_KW = {"vestas": 3600.0, "unison": 4200.0}
RD = 287.05
G = 9.80665
RHO0 = 1.225                                     # IEC 61400-12 기준 밀도
EPS = 1e-6

L = "ldaps_mean_"
F = "gfs_mean_"


def _g(df, col, default=None):
    return df[col].to_numpy(float) if col in df.columns else default


def _ws(u, v):
    return np.sqrt(u * u + v * v)


# ------------------------------------------------------------ 1. 허브고도 풍속
def add_hub_wind(df):
    """
    LDAPS 50m 복원 + GFS 3점 시어로 허브고도(117m) 풍속을 두 갈래로 만든다.
    두 값의 격차 자체가 불확실성 지표라 그것도 피처로 남긴다.
    """
    out = {}

    # --- LDAPS: 50m 평균바람 복원 (max/min 의 중점) ---
    u50 = _g(df, L + "heightAboveGround_50_50MUmax")
    u50n = _g(df, L + "heightAboveGround_50_50MUmin")
    v50 = _g(df, L + "heightAboveGround_50_50MVmax")
    v50n = _g(df, L + "heightAboveGround_50_50MVmin")
    u10 = _g(df, L + "heightAboveGround_10_10u")
    v10 = _g(df, L + "heightAboveGround_10_10v")

    ws117_ldaps = None
    if all(x is not None for x in [u50, u50n, v50, v50n, u10, v10]):
        u50m, v50m = (u50 + u50n) / 2.0, (v50 + v50n) / 2.0
        ws50 = _ws(u50m, v50m)
        ws10 = _ws(u10, v10)
        out["ldaps_ws50"] = ws50
        out["ldaps_ws10"] = ws10

        a = np.log((ws50 + EPS) / (ws10 + EPS)) / np.log(50.0 / 10.0)
        a = np.clip(a, -0.3, 1.0)
        out["ldaps_alpha_10_50"] = a
        ws117_ldaps = ws50 * (HUB / 50.0) ** a
        out["ldaps_v117"] = ws117_ldaps
        out["ldaps_wd117"] = np.degrees(np.arctan2(-u50m, -v50m)) % 360.0

        # 난류강도: range 를 풍속으로 정규화해야 의미가 있다 (range ≈ 4σ)
        rng = _ws(u50 - u50n, v50 - v50n)
        out["ldaps_TI50"] = (rng / 4.0) / (ws50 + EPS)

    # --- GFS: 10 / 80 / 100 m 3점 최소제곱 시어 + 곡률 ---
    g10 = (_g(df, F + "heightAboveGround_10_10u"), _g(df, F + "heightAboveGround_10_10v"))
    g80 = (_g(df, F + "heightAboveGround_80_u"), _g(df, F + "heightAboveGround_80_v"))
    g100 = (_g(df, F + "heightAboveGround_100_100u"), _g(df, F + "heightAboveGround_100_100v"))

    ws117_gfs = None
    if all(x is not None for pair in (g10, g80, g100) for x in pair):
        w = np.stack([_ws(*g10), _ws(*g80), _ws(*g100)], axis=1)          # (n, 3)
        z = np.log(np.array([10.0, 80.0, 100.0]))
        lw = np.log(w + EPS)
        zc = z - z.mean()
        a3 = (lw - lw.mean(1, keepdims=True)) @ zc / (zc @ zc)            # 최소제곱 기울기
        a3 = np.clip(a3, -0.3, 1.0)
        out["gfs_alpha_3pt"] = a3
        ws117_gfs = w[:, 2] * (HUB / 100.0) ** a3
        out["gfs_v117_3pt"] = ws117_gfs
        out["gfs_ws80"] = w[:, 1]

        # 프로파일 곡률: 로그-로그에서 2차항. 안정층/제트면 볼록해진다
        Z = np.vstack([np.ones(3), zc, zc ** 2]).T
        coef = np.linalg.lstsq(Z, lw.T, rcond=None)[0]                    # (3, n)
        out["gfs_shear_curv"] = coef[2]

        # 경계층 평균 바람 대비 (미사용 컬럼 활용)
        pu, pv = _g(df, F + "planetaryBoundaryLayer_0_u"), _g(df, F + "planetaryBoundaryLayer_0_v")
        if pu is not None and pv is not None:
            wpbl = _ws(pu, pv)
            out["gfs_pbl_ws"] = wpbl
            out["gfs_v117_over_pbl"] = ws117_gfs / (wpbl + EPS)

    # --- 두 갈래 앙상블과 격차 ---
    if ws117_ldaps is not None and ws117_gfs is not None:
        out["v117_ens"] = 0.5 * (ws117_ldaps + ws117_gfs)
        out["v117_diff"] = ws117_ldaps - ws117_gfs
        out["v117_absdiff"] = np.abs(out["v117_diff"])
        out["v117_ratio"] = ws117_ldaps / (ws117_gfs + EPS)
    return out


# ------------------------------------------------------------ 2. 로터 등가풍속
def _rews_weights(R, n=9):
    """로터 원판을 높이로 n등분하고 각 층의 현(chord) 길이로 가중."""
    dz = np.linspace(-R * 0.95, R * 0.95, n)
    w = 2.0 * np.sqrt(np.maximum(R ** 2 - dz ** 2, 0.0))
    return HUB + dz, w / w.sum()


def add_rews(df, base_cols=("v117_ens", "ldaps_v117", "gfs_v117_3pt"),
             alpha_col_map=None):
    """
    허브고도 한 점이 아니라 로터 스윕 면적(54~180m / 49~185m) 전체의
    에너지 등가풍속. 시어가 클 때 허브고도 값이 과대평가되는 것을 보정한다.
        v_eq = [ Σ w_i · v(z_i)^3 ]^(1/3),  v(z) = v_hub · (z/117)^alpha
    """
    out = {}
    alpha_col_map = alpha_col_map or {
        "v117_ens": "ldaps_alpha_10_50", "ldaps_v117": "ldaps_alpha_10_50",
        "gfs_v117_3pt": "gfs_alpha_3pt"}
    for maker, R in ROTOR_R.items():
        z, w = _rews_weights(R)
        ratio_pow = (z / HUB)                                   # (n,)
        for c in base_cols:
            if c not in df.columns:
                continue
            ac = alpha_col_map.get(c)
            if ac is None or ac not in df.columns:
                continue
            v = df[c].to_numpy(float)[:, None]
            a = df[ac].to_numpy(float)[:, None]
            prof = v * ratio_pow[None, :] ** a                  # (n_rows, n_slice)
            veq = (np.nansum(w[None, :] * prof ** 3, axis=1)) ** (1.0 / 3.0)
            out[f"rews_{maker}_{c}"] = veq
            out[f"rews_gain_{maker}_{c}"] = veq / (v[:, 0] + EPS)
    return out


# ------------------------------------------------------------ 3. 습윤공기 밀도
def add_air_density(df, wind_cols=("v117_ens", "ldaps_v117", "gfs_v117_3pt")):
    """
    기존은 건조공기 + 지상기압이다. 두 군데를 고친다.
      1) 가상온도 Tv = T(1 + 0.608q).  LDAPS `_2_q`(비습) 가 이미 있다.
      2) 허브고도 기압 보정 p117 = p_sfc·exp(-g·117/(Rd·Tv))  (약 -1.5%)
    그리고 IEC 61400-12 밀도보정 풍속 v·(ρ/1.225)^(1/3) 을 만든다.
    발전량은 밀도에 선형이지만 파워커브는 풍속의 함수이므로 이 형태가 표준이다.
    """
    out = {}
    for tag, tcol, pcol, qcol in [
        ("ldaps", L + "heightAboveGround_2_t", L + "surface_0_sp", L + "heightAboveGround_2_q"),
        ("gfs", F + "heightAboveGround_2_2t", F + "surface_0_sp", F + "heightAboveGround_2_2sh")]:
        T = _g(df, tcol); P = _g(df, pcol); Q = _g(df, qcol)
        if T is None or P is None:
            continue
        T = np.where(T < 100.0, T + 273.15, T)                 # °C 로 들어오면 K 로
        Tv = T * (1.0 + 0.608 * Q) if Q is not None else T
        p117 = P * np.exp(-G * HUB / (RD * Tv))
        rho = p117 / (RD * Tv)
        out[f"rho_{tag}_moist"] = rho
        out[f"Tv_{tag}"] = Tv

    keys = [k for k in ("rho_ldaps_moist", "rho_gfs_moist") if k in out]
    if keys:
        rho_m = np.mean([out[k] for k in keys], axis=0)
        out["rho_mean_moist"] = rho_m
        fac = (rho_m / RHO0) ** (1.0 / 3.0)
        out["density_corr_factor"] = fac
        for c in wind_cols:
            if c in df.columns:
                out[f"{c}_dcorr"] = df[c].to_numpy(float) * fac
    return out


# ------------------------------------------------------------ 4. 경험 파워커브
def load_power_curves(scada_dir="./data/scada_derived"):
    """step1_scada_check.py 가 저장한 기종별 경험 파워커브. 학습기간 정적 곡선."""
    from pathlib import Path
    cur = {}
    for m in ROTOR_R:
        p = Path(scada_dir) / f"power_curve_{m}.csv"
        if p.exists():
            d = pd.read_csv(p, encoding="utf-8-sig")
            cur[m] = (d["ws"].to_numpy(float), d["kwh_per_h"].to_numpy(float))
    return cur


def _fallback_curve(maker):
    """곡선 파일이 없을 때 쓰는 이론 곡선 (EDA 확인값: cut-in 3.5, 정격 ~11)."""
    ws = np.linspace(0, 30, 121)
    ci, vr, co = 3.5, 11.0, 25.0
    p = np.where(ws < ci, 0.0,
        np.where(ws < vr, (ws ** 3 - ci ** 3) / (vr ** 3 - ci ** 3),
        np.where(ws < co, 1.0, 0.0)))
    return ws, np.clip(p, 0, 1) * TURB_KW[maker]


def add_power_curve(df, curves=None, wind_cols=("v117_ens_dcorr", "v117_ens",
                                                "ldaps_v117_dcorr", "gfs_v117_3pt_dcorr")):
    """
    SCADA 로 적합한 실제 파워커브를 통과시킨다. 이게 배치1 의 핵심이다.
    v^3 는 정격(10~11 m/s) 아래에서만 맞고 그 위에서는 상수다.
    지금 features.py 에는 이 평탄화가 없어서 모델이 26k 행으로 S커브를 혼자 배우고 있다.
    """
    out = {}
    curves = curves or {}
    for maker in ROTOR_R:
        c, r = curves.get(maker) or _fallback_curve(maker)
        rated = TURB_KW[maker]
        for wc in wind_cols:
            if wc not in df.columns:
                continue
            v = df[wc].to_numpy(float)
            pc = np.interp(v, c, r, left=0.0, right=0.0) / rated       # 0~1 정규화
            out[f"pc_{maker}_{wc}"] = pc

        # 로터 등가풍속 버전도 (시어 큰 상황에서 차이가 난다)
        rc = f"rews_{maker}_v117_ens"
        if rc in df.columns:
            out[f"pc_{maker}_rews"] = np.interp(
                df[rc].to_numpy(float), c, r, left=0.0, right=0.0) / rated

    # 운전 구간 표시 — 어느 풍속 추정이 맞는지 미리 정하지 않고 둘 다 만든다
    for tag, col in [("ens", "v117_ens_dcorr"), ("ldaps", "ldaps_v117_dcorr")]:
        if col not in df.columns:
            continue
        v = df[col].to_numpy(float)
        out[f"dist_to_rated_{tag}"] = v - 11.0
        out[f"region_flag_{tag}"] = np.select(
            [v < 3.5, v < 11.0, v < 25.0], [0, 1, 2], default=3).astype(float)
        out[f"above_rated_{tag}"] = (v >= 11.0).astype(float)
    return out


# ------------------------------------------------------------ 통합
def add_batch1_features(df, scada_dir="./data/scada_derived", verbose=True):
    """기존 파이프라인 결과에 배치1 피처를 덧붙인다."""
    df = df.copy()
    n0 = df.shape[1]

    for step, fn in [("허브고도 풍속", add_hub_wind)]:
        new = fn(df)
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
        if verbose:
            print(f"   {step}: +{len(new)}개")

    for step, fn in [("로터 등가풍속", add_rews), ("습윤공기 밀도", add_air_density)]:
        new = fn(df)
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
        if verbose:
            print(f"   {step}: +{len(new)}개")

    curves = load_power_curves(scada_dir)
    if verbose:
        got = ", ".join(f"{k}({len(v[0])}점)" for k, v in curves.items()) or "없음 → 이론곡선 대체"
        print(f"   파워커브 파일: {got}")
    new = add_power_curve(df, curves)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    if verbose:
        print(f"   경험 파워커브: +{len(new)}개")
        print(f"   합계 {n0} → {df.shape[1]}개 (+{df.shape[1]-n0})")

    return df.replace([np.inf, -np.inf], np.nan)