import os
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils import seed_everything, calculate_metric
from src.features import build_full_feature_pipeline
from src.logger import log_experiment
from src.postprocessing import optimize_postprocessing, apply_postprocessing

# ==============================================================================
# 0. 학습 소요 시간 측정 시작
# ==============================================================================
start_time = time.time()

# ==============================================================================
# 1. Config 로드
# ==============================================================================
config_path = "./configs/config_v6.yaml"  

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

seed_everything(config["seed"])
save_dir = Path(config["data_paths"]["save_model_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. 데이터 로드 및 기상 데이터 피벗
# ==============================================================================
train_dir = Path(config["data_paths"]["train_dir"])
train_labels = pd.read_csv(train_dir / "train_labels.csv", encoding="utf-8-sig")
ldaps_train = pd.read_csv(train_dir / "ldaps_train.csv", encoding="utf-8-sig")
gfs_train = pd.read_csv(train_dir / "gfs_train.csv", encoding="utf-8-sig")

def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", "grid_id", *drop_cols}]
    
    # Grid 별 Pivot
    pivoted = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=value_cols)
    pivoted.columns = [f"{prefix}_g{col[1]}_{col[0]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    # Grid 요약 Mean
    agg_mean = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg_mean.columns = [f"{prefix}_mean_{c}" for c in agg_mean.columns]
    agg_mean = agg_mean.reset_index()
    
    return pivoted.merge(agg_mean, on="forecast_kst_dtm", how="inner")

print("🔄 [Data Processing] 기상 데이터 피벗 및 마스터 데이터셋 구축 중...")
train_weather = process_weather_data(ldaps_train, "ldaps").merge(
    process_weather_data(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
)

train_base = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
train_base["forecast_kst_dtm"] = pd.to_datetime(train_base["forecast_kst_dtm"])
df_full = train_base.merge(train_weather, on="forecast_kst_dtm", how="left")

# ==============================================================================
# 3. 통합 피처 엔지니어링 & 다중공선성/미사용 변수 Pruning
# ==============================================================================
print("🚀 [Feature Engineering] 6대 도메인 피처 생성 및 Pruning 적용 중...")
df_processed = build_full_feature_pipeline(df_full)

# 무한대(inf) 정제
df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

# 학습 피처 컬럼 식별 및 저장
exclude_cols = ["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month"] + config["targets"]
feature_cols = [c for c in df_processed.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

joblib.dump(feature_cols, save_dir / "feature_cols.pkl")
print(f"📦 [Feature Spec] 최종 정제된 {len(feature_cols)}개 피처 명단 저장 완료: '{save_dir / 'feature_cols.pkl'}'")

# ==============================================================================
# 4. GroupKFold Cross Validation (연-월 그룹화 시계열 검증)
# ==============================================================================
n_splits = config.get("n_splits", 5)
gkf = GroupKFold(n_splits=n_splits)

oof_pred_df = pd.DataFrame(index=df_processed.index, columns=config["targets"], dtype=float)
model_type = config.get("model_type", "XGBoost")

for target in config["targets"]:
    print(f"\n🌀 [Target: {target.upper()}] {n_splits}-Fold GroupKFold 학습 시작...")
    train_mask = df_processed[target].notna()
    
    sub_df = df_processed.loc[train_mask].copy()
    X_target = sub_df[feature_cols].reset_index(drop=True)
    y_target = sub_df[target].reset_index(drop=True)
    groups = pd.to_datetime(sub_df["forecast_kst_dtm"]).dt.to_period("M").reset_index(drop=True)

    oof_preds_target = np.zeros(len(sub_df))
    cap = config["capacity_kwh"][target]

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_target, y_target, groups)):
        X_tr, y_tr = X_target.iloc[train_idx], y_target.iloc[train_idx]
        X_val, y_val = X_target.iloc[val_idx], y_target.iloc[val_idx]

        model_params = config.get("model_params", {}).copy()

        if model_type == "XGBoost":
            # Early stopping 인자 안전하게 처리
            early_stopping_rounds = model_params.pop("early_stopping_rounds", 30)
            model = XGBRegressor(**model_params, early_stopping_rounds=early_stopping_rounds)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif model_type == "LightGBM":
            model = LGBMRegressor(**model_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
        elif model_type == "CatBoost":
            model = CatBoostRegressor(**model_params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        elif model_type == "RandomForest":
            model = RandomForestRegressor(**model_params)
            model.fit(X_tr, y_tr)
        else:
            raise ValueError(f"지원하지 않는 model_type입니다: {model_type}")

        # Fold별 모델 저장
        joblib.dump(model, save_dir / f"model_{target}_fold{fold}.pkl")

        # Validation 예측 및 상한선 클리핑
        val_pred = model.predict(X_val)
        oof_preds_target[val_idx] = np.clip(val_pred, 0, cap)

    # OOF 예측값 기록
    oof_pred_df.loc[train_mask, target] = oof_preds_target

# ==============================================================================
# 5. OOF 기반 평가 산식 계산 및 Nelder-Mead 후처리
# ==============================================================================
answer_df = df_processed[config["targets"]].copy()

# 후처리 전 Raw OOF 점수
raw_score, raw_1_nmae, raw_ficr = calculate_metric(answer_df, oof_pred_df)

# Nelder-Mead FICR 극대화 후처리 탐색 및 저장
print("\n⚙️ [Post-Processing] Nelder-Mead 기반 FICR 극대화 최적화 파라미터 탐색 중...")
post_params = optimize_postprocessing(answer_df, oof_pred_df)
joblib.dump(post_params, save_dir / "post_params.pkl")

# 후처리 적용
oof_pred_post_df = apply_postprocessing(oof_pred_df, post_params)
total_score, one_minus_nmae, ficr = calculate_metric(answer_df, oof_pred_post_df)

# 소요 시간 산출
execution_time_sec = time.time() - start_time

print("=" * 60)
print(f"📊 [{config.get('version', 'v7')}] Post-Processed OOF Result")
print(f" - Total Score : {raw_score:.4f} -> {total_score:.4f} (▲ {total_score - raw_score:+.4f})")
print(f" - 1 - NMAE    : {raw_1_nmae:.4f} -> {one_minus_nmae:.4f}")
print(f" - FICR        : {raw_ficr:.4f} -> {ficr:.4f}")
print(f" - Exec Time   : {execution_time_sec:.2f} 초")
print("=" * 60)

# ==============================================================================
# 6. experiment_log.xlsx 자동 로깅
# ==============================================================================
log_experiment(
    config=config,
    total_score=total_score,
    one_minus_nmae=one_minus_nmae,
    ficr=ficr,
    execution_time_sec=execution_time_sec,
    features_summary=config.get("features_summary", f"v7 6대 도메인 피처 ({len(feature_cols)}개) + Pruning"),
    notes=f"{n_splits}-Fold GroupKFold(월별) / Nelder-Mead 후처리 적용 (+{total_score - raw_score:.4f})"
)

print(f"✅ 학습 및 후처리 완료! 모든 아티팩트가 '{save_dir}'에 저장되었습니다.")