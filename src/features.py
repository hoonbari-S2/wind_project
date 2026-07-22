import numpy as np
import pandas as pd

def process_calendar_features(dt_series):
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    
    # 삼각함수 주기성 변환
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out

def add_wind_features(df):
    df = df.copy()
    new_cols = {}
    
    def calc_wind_dict(u_series, v_series, prefix):
        ws = np.sqrt(u_series**2 + v_series**2)
        new_cols[f'{prefix}_ws'] = ws
        new_cols[f'{prefix}_ws_cubed'] = ws ** 3
        
        # 풍향 sin/cos 변환
        rad = np.arctan2(v_series, u_series)
        new_cols[f'{prefix}_wd_sin'] = np.sin(rad)
        new_cols[f'{prefix}_wd_cos'] = np.cos(rad)
        return ws

    # 1. GFS Mean 80m & 100m 풍속/풍향 산출
    gfs_u80 = df.get('gfs_mean_heightAboveGround_80_u')
    gfs_v80 = df.get('gfs_mean_heightAboveGround_80_v')
    gfs_u100 = df.get('gfs_mean_heightAboveGround_100_100u')
    gfs_v100 = df.get('gfs_mean_heightAboveGround_100_100v')
    
    v80_gfs, v100_gfs = None, None
    if gfs_u80 is not None and gfs_v80 is not None:
        v80_gfs = calc_wind_dict(gfs_u80, gfs_v80, 'gfs_mean_80m')
    if gfs_u100 is not None and gfs_v100 is not None:
        v100_gfs = calc_wind_dict(gfs_u100, gfs_v100, 'gfs_mean_100m')
        
    # [v4] 117m Hub Height 멱법칙(Power Law) 보정 풍속 계산
    if v80_gfs is not None and v100_gfs is not None:
        v80_safe = np.maximum(v80_gfs, 0.01)
        v100_safe = np.maximum(v100_gfs, 0.01)
        
        # 80m와 100m 비율로 대기 연돌 지수(alpha) 계산
        alpha = np.log(v100_safe / v80_safe) / np.log(100.0 / 80.0)
        alpha_clipped = np.clip(alpha, -0.5, 1.0)
        
        # 117m 고도 풍속 외삽
        v117_gfs = v100_safe * ((117.0 / 100.0) ** alpha_clipped)
        new_cols['gfs_mean_117m_ws'] = v117_gfs
        new_cols['gfs_mean_117m_ws_cubed'] = v117_gfs ** 3
        new_cols['gfs_shear_alpha_100_80'] = alpha_clipped
        new_cols['gfs_shear_diff_100_80'] = v100_gfs - v80_gfs

    # 2. LDAPS Mean 10m -> 117m 산악 지형 멱법칙 적용
    ldaps_u10 = df.get('ldaps_mean_heightAboveGround_10_10u')
    ldaps_v10 = df.get('ldaps_mean_heightAboveGround_10_10v')
    if ldaps_u10 is not None and ldaps_v10 is not None:
        v10_ldaps = calc_wind_dict(ldaps_u10, ldaps_v10, 'ldaps_mean_10m')
        v10_safe = np.maximum(v10_ldaps, 0.01)
        
        # 산악 지역 표준 alpha = 0.20 적용
        v117_ldaps = v10_safe * ((117.0 / 10.0) ** 0.20)
        new_cols['ldaps_mean_117m_ws'] = v117_ldaps
        new_cols['ldaps_mean_117m_ws_cubed'] = v117_ldaps ** 3

    # 3. GFS vs LDAPS 117m 풍속 차이 피처
    if 'gfs_mean_117m_ws' in new_cols and 'ldaps_mean_117m_ws' in new_cols:
        new_cols['diff_ws_117m_gfs_ldaps'] = new_cols['gfs_mean_117m_ws'] - new_cols['ldaps_mean_117m_ws']

    # 4. 피벗된 개별 격자(Grid)별 풍속 및 sin/cos 자동 생성
    for col in list(df.columns):
        if col.endswith('_heightAboveGround_80_u'):
            prefix = col.replace('_heightAboveGround_80_u', '_80m')
            v_col = col.replace('_80_u', '_80_v')
            if v_col in df.columns:
                calc_wind_dict(df[col], df[v_col], prefix)
        elif col.endswith('_heightAboveGround_100_100u'):
            prefix = col.replace('_heightAboveGround_100_100u', '_100m')
            v_col = col.replace('_100_100u', '_100_100v')
            if v_col in df.columns:
                calc_wind_dict(df[col], df[v_col], prefix)
        elif col.endswith('_heightAboveGround_10_10u'):
            prefix = col.replace('_heightAboveGround_10_10u', '_10m')
            v_col = col.replace('_10_10u', '_10_10v')
            if v_col in df.columns:
                calc_wind_dict(df[col], df[v_col], prefix)

    # --- [v5 신규 추가] 4번 가설: 푄현상(높새바람) & 열돔/대기정체 피처 ---
    u_gfs = df.get('gfs_mean_heightAboveGround_100_100u', 0)
    easterly_wind = np.maximum(0, -u_gfs)  # 동풍 강도만 추출

    t_c = df.get('ldaps_mean_heightAboveGround_2_t', 20.0)
    rh = np.clip(df.get('ldaps_mean_heightAboveGround_2_r', 50.0), 1.0, 100.0)
    td = t_c - ((100.0 - rh) / 5.0)  # 이슬점 온도 간이 수식
    temp_dew_spread = np.maximum(0, t_c - td)  # 기온-이슬점 차이 (건조도)

    # 문자열 타입 파싱 대응 안전 장치
    if 'forecast_kst_dtm' in df.columns:
        dt_col = pd.to_datetime(df['forecast_kst_dtm'])
        month = dt_col.dt.month
        hour = dt_col.dt.hour
    else:
        month = df['month'] if 'month' in df.columns else 5
        hour = df['hour'] if 'hour' in df.columns else 12

    is_foehn_season = month.isin([4, 5, 6, 7]).astype(int) if isinstance(month, pd.Series) else (month in [4, 5, 6, 7])
    is_daytime = hour.between(10, 18).astype(int) if isinstance(hour, pd.Series) else (10 <= hour <= 18)

    # 높새바람 지수 (시간/계절 가중치 적용)
    new_cols['foehn_index_raw'] = easterly_wind * temp_dew_spread
    new_cols['foehn_index_seasonal'] = easterly_wind * temp_dew_spread * is_foehn_season
    new_cols['foehn_index_peak'] = easterly_wind * temp_dew_spread * is_foehn_season * is_daytime

    # 열돔 / 고온 무풍 정체 지수 (여름철 7~8월 고온 + 저풍속)
    ws_117m = new_cols.get('gfs_mean_117m_ws', 5.0)
    is_summer = month.isin([7, 8]).astype(int) if isinstance(month, pd.Series) else (month in [7, 8])
    new_cols['heat_stagnation_index'] = (t_c / (ws_117m + 0.1)) * is_summer

    # --- [v5 신규 추가] 5번 가설: 시계열 급변(Ramp Event) 및 이동 평균 ---
    if 'gfs_mean_117m_ws' in new_cols:
        ws = new_cols['gfs_mean_117m_ws']
        new_cols['ws_diff_prev1'] = ws.diff(1).fillna(0)   # t - (t-1)
        new_cols['ws_diff_next1'] = ws.diff(-1).fillna(0)  # t - (t+1)
        
        # 3시간 이동평균 및 이동표준편차 (바람의 변동성)
        new_cols['ws_rolling_mean_3h'] = ws.rolling(3, min_periods=1, center=True).mean()
        new_cols['ws_rolling_std_3h'] = ws.rolling(3, min_periods=1, center=True).std().fillna(0)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)