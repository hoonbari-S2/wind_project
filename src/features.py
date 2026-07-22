import numpy as np
import pandas as pd

def process_calendar_features(dt_series):
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    
    # 주기성 삼각함수 변환
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out

def add_wind_features(df):
    """ 
    [v2 피처 엔지니어링]
    1. GFS 80m, 100m 바람 성분(U, V) 합성 풍속 및 3제곱 변수 추가
    2. 풍향 (Wind Direction, Degree) 피처 추가
    """
    df = df.copy()
    
    # -------------------------------------------------------------
    # 1. GFS 80m 바람 피처 (기존 + 풍향 추가)
    # -------------------------------------------------------------
    if 'gfs_heightAboveGround_80_u_mean' in df.columns and 'gfs_heightAboveGround_80_v_mean' in df.columns:
        u80 = df['gfs_heightAboveGround_80_u_mean']
        v80 = df['gfs_heightAboveGround_80_v_mean']
        
        # 합성 풍속 & 풍속 3제곱
        df['gfs_wind_speed_80'] = np.sqrt(u80**2 + v80**2)
        df['gfs_wind_speed_80_cubed'] = df['gfs_wind_speed_80'] ** 3
        
        # 풍향 (0~360도)
        df['gfs_wind_dir_80'] = np.degrees(np.arctan2(v80, u80)) % 360

    # -------------------------------------------------------------
    # 2. GFS 100m 바람 피처 (v2 신규 추가)
    # -------------------------------------------------------------
    if 'gfs_heightAboveGround_100_u_mean' in df.columns and 'gfs_heightAboveGround_100_v_mean' in df.columns:
        u100 = df['gfs_heightAboveGround_100_u_mean']
        v100 = df['gfs_heightAboveGround_100_v_mean']
        
        # 합성 풍속 & 풍속 3제곱
        df['gfs_wind_speed_100'] = np.sqrt(u100**2 + v100**2)
        df['gfs_wind_speed_100_cubed'] = df['gfs_wind_speed_100'] ** 3
        
        # 풍향 (0~360도)
        df['gfs_wind_dir_100'] = np.degrees(np.arctan2(v100, u100)) % 360

    # -------------------------------------------------------------
    # 3. 80m-100m 고도 간 풍속 차이 (전단력/Shear 특성, v2 신규 추가)
    # -------------------------------------------------------------
    if 'gfs_wind_speed_80' in df.columns and 'gfs_wind_speed_100' in df.columns:
        df['gfs_wind_shear_100_80'] = df['gfs_wind_speed_100'] - df['gfs_wind_speed_80']

    return df