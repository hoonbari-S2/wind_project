# src/postprocessing.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils import calculate_metric, CAPACITY_KWH, TARGET_COLS

def apply_postprocessing(pred_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    최적화된 파라미터(alpha, beta)를 적용해 예측값을 보정하는 함수
    """
    processed_df = pred_df.copy()
    for col in TARGET_COLS:
        if col in params:
            alpha = params[col]["alpha"]
            beta = params[col]["beta"]
            cap = CAPACITY_KWH[col]
            
            # 1. 선형 스케일 보정
            adjusted = processed_df[col] * alpha + beta
            
            # 2. 물리적 한계값 클리핑 (0 ~ 설비용량)
            processed_df[col] = np.clip(adjusted, 0, cap)
            
    return processed_df

def optimize_postprocessing(answer_df: pd.DataFrame, oof_pred_df: pd.DataFrame) -> dict:
    """
    OOF 예측값을 바탕으로 Total Score를 극대화하는 그룹별 최적 보정 파라미터 탐색
    """
    best_params = {}
    
    # Valid 라벨이 존재하는 구간만 추출하여 최적화
    for col in TARGET_COLS:
        valid_mask = answer_df[col].notna()
        y_true_col = answer_df.loc[valid_mask, [col]]
        y_pred_col = oof_pred_df.loc[valid_mask, [col]]
        cap = CAPACITY_KWH[col]

        def objective(x):
            alpha, beta = x
            temp_pred = y_pred_col.copy()
            temp_pred[col] = np.clip(temp_pred[col] * alpha + beta, 0, cap)
            
            # 단일 그룹에 대한 Score 산출 (음수화하여 minimize 적용)
            # utils.calculate_metric 구성을 단일 컬럼에 대해 개별 적용
            actual = y_true_col[col].to_numpy(dtype=float)
            forecast = temp_pred[col].to_numpy(dtype=float)
            
            valid = actual >= cap * 0.10
            act_v = actual[valid]
            fore_v = forecast[valid]
            
            if len(act_v) == 0:
                return 0.0
                
            err = np.abs(fore_v - act_v) / cap
            nmae = np.mean(err)
            
            unit_price = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
            ficr = np.sum(act_v * unit_price) / np.sum(act_v * 4.0)
            
            score = 0.5 * (1 - nmae) + 0.5 * ficr
            return -score  # Total Score 극대화

        # 초기값: alpha=1.0 (그대로), beta=0.0 (지프시프트 없음)
        res = minimize(objective, [1.0, 0.0], method='Nelder-Mead')
        best_params[col] = {"alpha": res.x[0], "beta": res.x[1]}
        
    return best_params