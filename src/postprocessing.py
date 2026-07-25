"""
==============================================================================
제 3회 풍력발전량 예측 AI 경진대회 - 후처리(Post-processing) 모듈

[1. 최적화 보정 수식]
  Post-processed Prediction = Clip(Where(alpha * y_pred + beta < zero_th, 0.0, alpha * y_pred + beta), 0, Capacity)

[2. 최적화 파라미터의 도메인 의미]
  1) alpha (Slope/Scale Factor) : 모델의 전반적 과대/과소 예측 경향성 교정 (기울기 보정)
  2) beta (Intercept/Bias Shift) : 센서 오차 및 대기 마찰 등에 의한 전체적 편향 이동 (절편 보정)
  3) zero_th (Zero Threshold)    : Cut-in 미달 저풍속 시 발생되는 미세 노이즈를 0 kWh로 강제 절단 (제로 팽창 분포 모사)

[3. Nelder-Mead 최적화 알고리즘 채택 이유]
  - 대회 평가 산식(Total Score)인 FICR(구간별 유닛 가격 $4, $3, $0 계단식 적용) 및 np.where 절단 로직은 미분 불가능한 불연속 함수임.
  - 경사하강법(Gradient Descent)은 기울기(Gradient)가 0이 되거나 수렴에 실패하므로, 
    미분을 사용하지 않는 비구배(Non-gradient) 다면체 탐색 알고리즘인 Nelder-Mead를 적용하여 대회 산식 점수를 직접 극대화(Maximize)함.
==============================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils import calculate_metric, CAPACITY_KWH, TARGET_COLS


def apply_postprocessing(pred_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    최적화된 파라미터(alpha, beta, zero_threshold)를 적용해 예측값을 보정하는 함수
    """
    processed_df = pred_df.copy()
    for col in TARGET_COLS:
        if col in params:
            alpha = params[col].get("alpha", 1.0)
            beta = params[col].get("beta", 0.0)
            zero_th = params[col].get("zero_th", 0.0)
            cap = CAPACITY_KWH[col]
            
            # 1. 선형 스케일 보정
            adjusted = processed_df[col] * alpha + beta
            
            # 2. 저풍속/미세 노이즈 0 kWh 절단 보정
            adjusted = np.where(adjusted < zero_th, 0.0, adjusted)
            
            # 3. 설비용량 상하한 클리핑 (0 ~ Group Capacity)
            processed_df[col] = np.clip(adjusted, 0.0, cap)
            
    return processed_df


def optimize_postprocessing(answer_df: pd.DataFrame, oof_pred_df: pd.DataFrame) -> dict:
    """
    OOF 예측값을 바탕으로 대회 평가 산식(Total Score)을 극대화하는 그룹별 최적 보정 파라미터 탐색
    """
    best_params = {}
    
    for col in TARGET_COLS: # 'kpx_group_1', 'kpx_group_2', 'kpx_group_3'을 순회
        valid_mask = answer_df[col].notna()
        y_true_col = answer_df.loc[valid_mask, col].to_numpy(dtype=float)
        y_pred_col = oof_pred_df.loc[valid_mask, col].to_numpy(dtype=float)
        cap = CAPACITY_KWH[col]

        def objective(x):
            alpha, beta, zero_th = x
            
            # 보정 연산
            adjusted = y_pred_col * alpha + beta
            adjusted = np.where(adjusted < zero_th, 0.0, adjusted)
            forecast = np.clip(adjusted, 0.0, cap)
            
            # 대회 공식 평가 산식 계산 (FICR 10% 조건 반영)
            valid = y_true_col >= cap * 0.10
            act_v = y_true_col[valid]
            fore_v = forecast[valid]
            
            if len(act_v) == 0:
                return 0.0
                
            err = np.abs(fore_v - act_v) / cap
            nmae = np.mean(err)
            
            unit_price = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
            ficr = np.sum(act_v * unit_price) / np.sum(act_v * 4.0)
            
            score = 0.5 * (1.0 - nmae) + 0.5 * ficr
            return -score  # minimize를 위해 음수화

        # 초기값: alpha=1.0, beta=0.0, zero_th=10.0 (10 kWh 미만 절단)
        initial_guess = [1.0, 0.0, 10.0]
        res = minimize(objective, initial_guess, method='Nelder-Mead')
        # col(타겟 그룹) 단위로 y_true, y_pred, cap을 따로 추출하여 최적화 실행
        best_params[col] = {
            "alpha": res.x[0],
            "beta": res.x[1],
            "zero_th": max(0.0, res.x[2])  # 임계값은 양수 보장
        }
        
    return best_params