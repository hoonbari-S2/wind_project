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
    """
    [v4 피처 엔지니어링]
    1. GFS 80m & 100m 데이터를 이용한 117m 허브 높이 멱법칙(Power Law) 풍속 도출
    2. LDAPS 10m 데이터 산악 지형 멱법칙 적용 117m 풍속 도출
    3. 풍향(Wind Direction) 삼각함수(sin/cos) 연속 변환
    4. GFS vs LDAPS 117m 풍속 예측 차이 (변동성 지표)
    """
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
        
    # [핵심] 117m Hub Height 멱법칙(Power Law) 보정 풍속 계산
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

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)