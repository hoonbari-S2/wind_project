"""
src/features.py
==============================================================================
제 3회 풍력발전량 예측 AI 경진대회 - 피처 엔지니어링 파이프라인
- EDA 검증 물리 수식, 열역학 공기밀도, WPD, 푄현상, 커테일먼트 반영
- 다중공선성 정제 (공간/물리 수식 중복 1차 단순화 & 미사용 피처 가지치기)
==============================================================================
"""

import numpy as np
import pandas as pd


def process_calendar_features(dt_series):
    """시공간 달력 변수 및 삼각함수(Sin/Cos) 주기성 변수 생성"""
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    
    out["month"] = dt.dt.month
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    
    # 24시간 및 12개월 삼각함수 주기성 변환
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    
    return out


def add_weather_uncertainty_features(df):
    """1. 예보 모델 간 불확실성 및 예보 격차 피처 (LDAPS vs GFS 공통 칼럼)"""
    df = df.copy()
    new_cols = {}
    
    u_ldaps_10 = df.get('ldaps_mean_heightAboveGround_10_10u')
    v_ldaps_10 = df.get('ldaps_mean_heightAboveGround_10_10v')
    u_gfs_10 = df.get('gfs_mean_heightAboveGround_10_10u')
    v_gfs_10 = df.get('gfs_mean_heightAboveGround_10_10v')
    
    if all(col is not None for col in [u_ldaps_10, v_ldaps_10, u_gfs_10, v_gfs_10]):
        ws_ldaps_10 = np.sqrt(u_ldaps_10**2 + v_ldaps_10**2)
        ws_gfs_10 = np.sqrt(u_gfs_10**2 + v_gfs_10**2)
        
        new_cols['diff_10u_ldaps_gfs'] = u_ldaps_10 - u_gfs_10
        new_cols['diff_10v_ldaps_gfs'] = v_ldaps_10 - v_gfs_10
        new_cols['diff_ws10_ldaps_gfs'] = ws_ldaps_10 - ws_gfs_10
        new_cols['abs_diff_ws10_ldaps_gfs'] = np.abs(ws_ldaps_10 - ws_gfs_10)

    # 기압 편차 (surface_0_sp, meanSea_0_prmsl)
    p_surf_ldaps = df.get('ldaps_mean_surface_0_sp')
    p_surf_gfs = df.get('gfs_mean_surface_0_sp')
    if p_surf_ldaps is not None and p_surf_gfs is not None:
        new_cols['diff_p_surf_ldaps_gfs'] = p_surf_ldaps - p_surf_gfs
        new_cols['abs_diff_p_surf_ldaps_gfs'] = np.abs(new_cols['diff_p_surf_ldaps_gfs'])

    p_msl_ldaps = df.get('ldaps_mean_meanSea_0_prmsl')
    p_msl_gfs = df.get('gfs_mean_meanSea_0_prmsl')
    if p_msl_ldaps is not None and p_msl_gfs is not None:
        new_cols['diff_p_msl_ldaps_gfs'] = p_msl_ldaps - p_msl_gfs
        new_cols['abs_diff_p_msl_ldaps_gfs'] = np.abs(new_cols['diff_p_msl_ldaps_gfs'])

    # 기온 및 이슬점, 건조도 격차
    t2m_ldaps = df.get('ldaps_mean_heightAboveGround_2_t')
    t2m_gfs = df.get('gfs_mean_heightAboveGround_2_2t', df.get('gfs_mean_heightAboveGround_2_t'))
    d2m_ldaps = df.get('ldaps_mean_heightAboveGround_2_dpt')
    d2m_gfs = df.get('gfs_mean_heightAboveGround_2_2d')

    if all(col is not None for col in [t2m_ldaps, t2m_gfs, d2m_ldaps, d2m_gfs]):
        new_cols['diff_t2m_ldaps_gfs'] = t2m_ldaps - t2m_gfs
        new_cols['abs_diff_t2m_ldaps_gfs'] = np.abs(new_cols['diff_t2m_ldaps_gfs'])
        new_cols['diff_d2m_ldaps_gfs'] = d2m_ldaps - d2m_gfs
        new_cols['abs_diff_d2m_ldaps_gfs'] = np.abs(new_cols['diff_d2m_ldaps_gfs'])
        
        dew_dep_ldaps = t2m_ldaps - d2m_ldaps
        dew_dep_gfs = t2m_gfs - d2m_gfs
        new_cols['diff_dew_depression_ldaps_gfs'] = dew_dep_ldaps - dew_dep_gfs

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def add_upper_air_and_shear_features(df):
    """2. 상층 대기 연직 유입, 돌풍, 멱법칙 117m 풍속 및 시계열 차분 피처 (GFS 고유)"""
    df = df.copy()
    new_cols = {}
    eps = 1e-5

    u_gfs_10 = df.get('gfs_mean_heightAboveGround_10_10u')
    v_gfs_10 = df.get('gfs_mean_heightAboveGround_10_10v')
    u_gfs_100 = df.get('gfs_mean_heightAboveGround_100_100u')
    v_gfs_100 = df.get('gfs_mean_heightAboveGround_100_100v')
    u_gfs_850 = df.get('gfs_mean_isobaricInhPa_850_u')
    v_gfs_850 = df.get('gfs_mean_isobaricInhPa_850_v')

    if all(col is not None for col in [u_gfs_10, v_gfs_10, u_gfs_100, v_gfs_100, u_gfs_850, v_gfs_850]):
        ws_gfs_10 = np.sqrt(u_gfs_10**2 + v_gfs_10**2)
        ws_gfs_100 = np.sqrt(u_gfs_100**2 + v_gfs_100**2)
        ws_gfs_850 = np.sqrt(u_gfs_850**2 + v_gfs_850**2)

        # 850hPa 상공 바람 하부 유입 비율
        new_cols['gfs_v850_inflow_ratio'] = ws_gfs_100 / (ws_gfs_850 + eps)

        # 850hPa 연직 기온 감률
        t2m_gfs = df.get('gfs_mean_heightAboveGround_2_2t', df.get('gfs_mean_heightAboveGround_2_t'))
        t850_gfs = df.get('gfs_mean_isobaricInhPa_850_t')
        if t2m_gfs is not None and t850_gfs is not None:
            new_cols['gfs_lapse_rate_850'] = (t2m_gfs - t850_gfs) / 1.45

        # 돌풍 지수
        gust_gfs = df.get('gfs_mean_surface_0_gust')
        if gust_gfs is not None:
            new_cols['gfs_gust_ratio_10m'] = gust_gfs / (ws_gfs_10 + eps)
            new_cols['gfs_gust_ratio_100m'] = gust_gfs / (ws_gfs_100 + eps)

        # 117m 멱법칙(Power Law) 외삽 풍속
        alpha_gfs = np.log((ws_gfs_100 + eps) / (ws_gfs_10 + eps)) / np.log(100.0 / 10.0)
        alpha_clipped = np.clip(alpha_gfs, -0.5, 1.0)
        new_cols['gfs_alpha_shear'] = alpha_clipped
        
        v117_gfs = ws_gfs_100 * ((117.0 / 100.0) ** alpha_clipped)
        new_cols['gfs_v117_powerlaw'] = v117_gfs

        # 시계열 Ramp Event, Lag, Rolling 피처
        new_cols['gfs_v117_ramp_1h'] = v117_gfs.diff(1).fillna(0)
        new_cols['gfs_v117_ramp_2h'] = v117_gfs.diff(2).fillna(0)
        new_cols['gfs_v117_accel'] = new_cols['gfs_v117_ramp_1h'].diff(1).fillna(0)
        new_cols['gfs_v117_lag1'] = v117_gfs.shift(1).bfill()
        new_cols['gfs_v117_lag2'] = v117_gfs.shift(2).bfill()
        new_cols['gfs_v117_roll_mean_3h'] = v117_gfs.rolling(3, min_periods=1, center=False).mean()
        new_cols['gfs_v117_roll_std_3h'] = v117_gfs.rolling(3, min_periods=1, center=False).std().fillna(0)
        
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def add_ldaps_turbulence_and_radiation_features(df):
    """3. 미세 지형 난류, 경계층 높이(BLH), 복사 및 산곡풍 특성 피처 (LDAPS 고유)"""
    df = df.copy()
    new_cols = {}
    eps = 1e-5

    u50_max = df.get('ldaps_mean_heightAboveGround_50_50MUmax')
    u50_min = df.get('ldaps_mean_heightAboveGround_50_50MUmin')
    v50_max = df.get('ldaps_mean_heightAboveGround_50_50MVmax')
    v50_min = df.get('ldaps_mean_heightAboveGround_50_50MVmin')

    if all(col is not None for col in [u50_max, u50_min, v50_max, v50_min]):
        u50_range = u50_max - u50_min
        v50_range = v50_max - v50_min
        new_cols['ldaps_u50_turb_range'] = u50_range
        new_cols['ldaps_v50_turb_range'] = v50_range
        new_cols['ldaps_ws50_turb_range'] = np.sqrt(u50_range**2 + v50_range**2)

    blh = df.get('ldaps_mean_etc_0_blh')
    if blh is not None:
        new_cols['ldaps_blh_inv'] = 1.0 / (blh + eps)
        v_hub = df.get('gfs_v117_powerlaw')
        if v_hub is None:
            u10 = df.get('ldaps_mean_heightAboveGround_10_10u', 0)
            v10 = df.get('ldaps_mean_heightAboveGround_10_10v', 0)
            v_hub = np.sqrt(u10**2 + v10**2)
        new_cols['ldaps_blh_v117_interaction'] = (v_hub / (blh + eps)) * 1000.0

    sw_dir = df.get('ldaps_mean_heightAboveGround_2_SWDIR')
    sw_dif = df.get('ldaps_mean_heightAboveGround_2_SWDIF')
    if sw_dir is not None and sw_dif is not None:
        sw_total = sw_dir + sw_dif
        new_cols['ldaps_sw_total'] = sw_total
        new_cols['ldaps_sw_dir_ratio'] = sw_dir / (sw_total + eps)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def add_thermodynamic_and_wpd_features(df):
    """4. 이상기체 상태방정식 공기 밀도(rho) 및 풍력 에너지 밀도(WPD) 피처"""
    df = df.copy()
    new_cols = {}
    R_SPECIFIC = 287.05  # J / (kg · K)

    p_ldaps = df.get('ldaps_mean_surface_0_sp')
    t_ldaps = df.get('ldaps_mean_heightAboveGround_2_t')
    if p_ldaps is not None and t_ldaps is not None:
        t_ldaps_k = np.where(t_ldaps < 100.0, t_ldaps + 273.15, t_ldaps)
        rho_ldaps = p_ldaps / (R_SPECIFIC * t_ldaps_k)
        new_cols['rho_ldaps'] = rho_ldaps

    p_gfs = df.get('gfs_mean_surface_0_sp')
    t_gfs = df.get('gfs_mean_heightAboveGround_2_2t', df.get('gfs_mean_heightAboveGround_2_t'))
    if p_gfs is not None and t_gfs is not None:
        t_gfs_k = np.where(t_gfs < 100.0, t_gfs + 273.15, t_gfs)
        rho_gfs = p_gfs / (R_SPECIFIC * t_gfs_k)
        new_cols['rho_gfs'] = rho_gfs

    if 'rho_ldaps' in new_cols and 'rho_gfs' in new_cols:
        rho_mean = (new_cols['rho_ldaps'] + new_cols['rho_gfs']) / 2.0
        new_cols['rho_mean'] = rho_mean

    # WPD = 0.5 * rho * v^3
    u_ldaps_10 = df.get('ldaps_mean_heightAboveGround_10_10u')
    v_ldaps_10 = df.get('ldaps_mean_heightAboveGround_10_10v')
    if u_ldaps_10 is not None and v_ldaps_10 is not None and 'rho_ldaps' in new_cols:
        ws_ldaps_10 = np.sqrt(u_ldaps_10**2 + v_ldaps_10**2)
        new_cols['wpd_ldaps_10m'] = 0.5 * new_cols['rho_ldaps'] * (ws_ldaps_10 ** 3)

    ws_gfs_hub = df.get('gfs_v117_powerlaw')
    if ws_gfs_hub is not None and 'rho_gfs' in new_cols:
        new_cols['wpd_gfs_117m'] = 0.5 * new_cols['rho_gfs'] * (ws_gfs_hub ** 3)
        if 'rho_mean' in new_cols:
            new_cols['wpd_ensemble_117m'] = 0.5 * new_cols['rho_mean'] * (ws_gfs_hub ** 3)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def add_domain_turbine_and_foehn_features(df):
    """5 & 6. 터빈 제어(Cut-in/out, De-rating), 태백산맥 지형 분해, 착빙, 커테일먼트, 푄현상 및 열돔 지수"""
    df = df.copy()
    new_cols = {}

    v_hub = df.get('gfs_v117_powerlaw')
    if v_hub is None:
        u10 = df.get('ldaps_mean_heightAboveGround_10_10u', 0)
        v10 = df.get('ldaps_mean_heightAboveGround_10_10v', 0)
        v_hub = np.sqrt(u10**2 + v10**2)

    # 1) 터빈 작동 플래그 및 Soft Cut-out (De-rating)
    new_cols['turbine_active_flag'] = ((v_hub >= 3.5) & (v_hub < 25.0)).astype(int)
    
    def calc_soft_cutout_factor(v):
        if v < 20.0:
            return 1.0
        elif v >= 25.0:
            return 0.1
        else:
            return 1.0 - 0.9 * ((v - 20.0) / 5.0)

    soft_factor = v_hub.apply(calc_soft_cutout_factor)
    new_cols['soft_cutout_factor'] = soft_factor
    new_cols['effective_v117_derated'] = v_hub * soft_factor

    # 2) Cut-out 이력 현상 (Hysteresis)
    high_wind_past_3h = (v_hub >= 20.0).rolling(3, min_periods=1).max().shift(1).fillna(0)
    new_cols['cutout_hysteresis_flag'] = ((v_hub < 20.0) & (high_wind_past_3h == 1)).astype(int)

    # 3) 태백산맥 능선(약 170도 축) 직교/평행 바람 성분 분해
    u_10 = df.get('ldaps_mean_heightAboveGround_10_10u')
    v_10 = df.get('ldaps_mean_heightAboveGround_10_10v')
    if u_10 is not None and v_10 is not None:
        wd_rad = np.arctan2(-u_10, -v_10)
        new_cols['wind_dir_deg'] = (np.degrees(wd_rad)) % 360.0
        
        ridge_angle_rad = np.radians(170.0)
        new_cols['wind_perp_ridge'] = u_10 * np.cos(ridge_angle_rad) - v_10 * np.sin(ridge_angle_rad)
        new_cols['wind_parallel_ridge'] = u_10 * np.sin(ridge_angle_rad) + v_10 * np.cos(ridge_angle_rad)

    # 4) 착빙 위험도 (Blade Icing Risk)
    t_ldaps = df.get('ldaps_mean_heightAboveGround_2_t')
    dpt_ldaps = df.get('ldaps_mean_heightAboveGround_2_dpt')
    if t_ldaps is not None and dpt_ldaps is not None:
        t_celsius = np.where(t_ldaps > 100.0, t_ldaps - 273.15, t_ldaps)
        dew_dep = t_ldaps - dpt_ldaps
        new_cols['blade_icing_risk_flag'] = ((t_celsius <= 2.0) & (dew_dep <= 2.0)).astype(int)

    # 5) 시간/계절 파싱 및 커테일먼트, 푄현상, 열돔 지수
    if 'forecast_kst_dtm' in df.columns:
        dt_col = pd.to_datetime(df['forecast_kst_dtm'])
        month = dt_col.dt.month
        hour = dt_col.dt.hour
    else:
        month = df['month'] if 'month' in df.columns else 5
        hour = df['hour'] if 'hour' in df.columns else 12

    new_cols['curtailment_risk_hour'] = hour.isin([11, 12, 13, 14, 15]).astype(int)

    # 푄현상 (높새바람) 지수
    if u_10 is not None and t_ldaps is not None and dpt_ldaps is not None:
        easterly_wind = np.maximum(0.0, -u_10)
        temp_dew_spread = np.maximum(0.0, t_ldaps - dpt_ldaps)
        is_foehn_season = month.isin([4, 5, 6]).astype(int)
        is_daytime = hour.isin(range(9, 19)).astype(int)

        new_cols['foehn_index_raw'] = easterly_wind * temp_dew_spread
        new_cols['foehn_index_seasonal'] = new_cols['foehn_index_raw'] * is_foehn_season
        new_cols['foehn_index_peak'] = new_cols['foehn_index_seasonal'] * is_daytime

    # 열돔 무풍 정체 지수
    if t_ldaps is not None:
        t_c = np.where(t_ldaps > 100.0, t_ldaps - 273.15, t_ldaps)
        is_summer = month.isin([7, 8]).astype(int)
        new_cols['heat_stagnation_index'] = (np.maximum(0.0, t_c) / (v_hub + 0.1)) * is_summer

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def prune_redundant_and_zero_gain_features(df):
    """
    [다중공선성 & TreeSHAP 정제]
    1. 공간 중복 기압 격자(g1~g16) 제거 (전체 평균 컬럼 보존)
    2. 물리 수식 100% 중복(강수/강설) 컬럼 제거
    3. OOF TreeSHAP 검정 결과 Gain=0 & SHAP=0 완전 미사용 35개 피처 삭제
    """
    df = df.copy()

    # 1) 공간 중복 거시 기압 격자 제거
    macro_keywords = ['meanSea_0_prmsl', 'surface_0_sp']
    grid_macro_drops = [
        col for col in df.columns 
        if any(m in col for m in macro_keywords) and ('_g' in col) and ('_mean_' not in col)
    ]

    # 2) 물리 수식 100% 중복 변수 제거
    formula_dup_drops = [
        col for col in df.columns 
        if ('_avg_lsprate' in col) or ('_lssrate' in col)
    ]

    # 3) EDA 검정 미사용 (Gain=0 & SHAP=0) 35개 정적/중복 피처 명단
    zero_gain_shap_35 = [
        'ldaps_mean_surface_0_lsm', 'ldaps_mean_surface_0_h',
        'gfs_mean_surface_0_lsm', 'gfs_mean_surface_0_h',
        'ldaps_g1_surface_0_lsm', 'ldaps_g2_surface_0_lsm', 'ldaps_g3_surface_0_lsm',
        'ldaps_g4_surface_0_lsm', 'ldaps_g5_surface_0_lsm', 'ldaps_g6_surface_0_lsm',
        'ldaps_g7_surface_0_lsm', 'ldaps_g8_surface_0_lsm', 'ldaps_g9_surface_0_lsm',
        'ldaps_g10_surface_0_lsm', 'ldaps_g11_surface_0_lsm', 'ldaps_g12_surface_0_lsm',
        'ldaps_g13_surface_0_lsm', 'ldaps_g14_surface_0_lsm', 'ldaps_g15_surface_0_lsm',
        'ldaps_g16_surface_0_lsm', 'gfs_g1_surface_0_lsm', 'gfs_g2_surface_0_lsm',
        'gfs_g3_surface_0_lsm', 'gfs_g4_surface_0_lsm', 'gfs_g5_surface_0_lsm',
        'gfs_g6_surface_0_lsm', 'gfs_g7_surface_0_lsm', 'gfs_g8_surface_0_lsm',
        'gfs_g9_surface_0_lsm', 'ldaps_mean_surface_0_ncpcp', 'ldaps_mean_surface_0_snol',
        'gfs_mean_surface_0_tp', 'gfs_mean_surface_0_prate', 'diff_10u_ldaps_gfs',
        'diff_10v_ldaps_gfs'
    ]

    total_drops = list(set(grid_macro_drops + formula_dup_drops + zero_gain_shap_35))
    actual_drops = [c for c in total_drops if c in df.columns]

    return df.drop(columns=actual_drops)


def build_full_feature_pipeline(df):
    """
    [최종 통합 피처 엔지니어링 파이프라인]
    전체 파생 피처 생성 후 다중공선성/미사용 변수까지 일괄 정제하여 반환
    """
    df = df.copy()
    
    # 1. 달력 시공간 피처
    cal_df = process_calendar_features(df['forecast_kst_dtm'])
    df = pd.concat([df, cal_df], axis=1)

    # 2. 6대 도메인 파생 피처 순차 결합
    df = add_weather_uncertainty_features(df)
    df = add_upper_air_and_shear_features(df)
    df = add_ldaps_turbulence_and_radiation_features(df)
    df = add_thermodynamic_and_wpd_features(df)
    df = add_domain_turbine_and_foehn_features(df)

    # 3. 다중공선성 및 TreeSHAP 미사용 변수 가지치기
    df = prune_redundant_and_zero_gain_features(df)

    return df