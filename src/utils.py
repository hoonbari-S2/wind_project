import os
import random
import numpy as np
import pandas as pd

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def calculate_metric(answer_df: pd.DataFrame, pred_df: pd.DataFrame):
    """
    데이콘 공식 평가 지표 계산 함수
    
    Returns:
        total_score (float): 0.5 * (1-NMAE) + 0.5 * FICR
        one_minus_nmae (float): 1 - NMAE
        ficr (float): FICR 정산금 획득률
    """
    group_nmae = []
    group_ficr = []

    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]

        # 실제 발전량이 설비용량의 10% 이상인 시간대만 평가
        valid = actual >= capacity * 0.10

        actual = actual[valid]
        forecast = forecast[valid]

        if len(actual) == 0:
            group_nmae.append(0.0)
            group_ficr.append(0.0)
            continue

        # NMAE 계산
        error_rate = np.abs(forecast - actual) / capacity
        group_nmae.append(np.mean(error_rate))

        # FICR 계산 (발전량 가중 정산금)
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )

        earned_settlement = np.sum(actual * unit_price)
        max_settlement = np.sum(actual * 4.0)

        group_ficr.append(earned_settlement / max_settlement)

    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)

    total_score = 0.5 * one_minus_nmae + 0.5 * ficr

    return total_score, one_minus_nmae, ficr