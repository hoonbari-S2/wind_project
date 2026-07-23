import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils import seed_everything, calculate_metric
from src.features import process_calendar_features, add_wind_features
from src.logger import log_experiment
from src.postprocessing import optimize_postprocessing, apply_postprocessing

# 1. Config 로드
config_path = "./configs/config_v5.yaml"  # 필요 시 config_v6.yaml 등으로 변경 가능
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

seed_everything(config["seed"])
save_dir = Path(config["data_paths"]["save_model_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

# 2. 데이터 로드
train_dir = Path(config["data_paths"]["train_dir"])
train_labels = pd.read_csv(train_dir / "train_labels.csv", encoding="utf-8-sig")
ldaps_train = pd.read_csv(train_dir / "ldaps_train.csv", encoding="utf-8-sig")
gfs_train = pd.read_csv(train_dir / "gfs_train.csv", encoding="utf-8-sig")

def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", "grid_id", *drop_cols}]
    
    # 1. Grid 별 Pivot
    pivoted = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=value_cols)
    pivoted.columns = [f"{prefix}_g{col[1]}_{col[0]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    # 2. Grid 요약 Mean
    agg_mean = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg_mean.columns = [f"{prefix}_mean_{c}" for c in agg_mean.columns]
    agg_mean = agg_mean.reset_index()
    
    return pivoted.merge(agg_mean, on="forecast_kst_dtm", how="inner")

print("🔄 [Data Processing] 기상 데이터 피벗 및 전처리 진행 중...")
train_weather = process_weather_data(ldaps_train, "ldaps").merge(
    process_weather_data(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
)

train_base = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
train_base["forecast_kst_dtm"] = pd.to_datetime(train_base["forecast_kst_dtm"])
train_df = train_base.merge(train_weather, on="forecast_kst_dtm", how="left")

# 3. 피처 생성 및 정제
cal_feat = process_calendar_features(train_df["forecast_kst_dtm"])
X_train_raw = pd.concat(
    [cal_feat, train_df.drop(columns=["forecast_kst_dtm", *config["targets"]])],
    axis=1
)
X_train_raw = add_wind_features(X_train_raw)

# 무한대(inf) 값을 np.nan으로 정제
X_train_imp = X_train_raw.replace([np.inf, -np.inf], np.nan)

# 피처 컬럼명 리스트 저장 (추론 시 컬럼 순서/이름 일치 보장)
feature_cols = list(X_train_imp.columns)
joblib.dump(feature_cols, save_dir / "feature_cols.pkl")
print(f"📦 [Feature Spec] 총 {len(feature_cols)}개 피처 리스트 저장 완료: '{save_dir / 'feature_cols.pkl'}'")

# 4. GroupKFold Cross Validation (연-월 그룹화로 시계열 Data Leakage 방지)
n_splits = config.get("n_splits", 5)
gkf = GroupKFold(n_splits=n_splits)

oof_pred_df = pd.DataFrame(index=train_df.index, columns=config["targets"], dtype=float)
model_type = config.get("model_type", "LightGBM")

for target in config["targets"]:
    print(f"\n🌀 [Target: {target}] {n_splits}-Fold GroupKFold Cross Validation 진행 중...")
    train_mask = train_df[target].notna()
    
    # Target 데이터 및 Group 지정 (연-월 기준 그룹화)
    sub_train_df = train_df.loc[train_mask].copy()
    X_target = X_train_imp.loc[train_mask][feature_cols]
    y_target = sub_train_df[target]
    groups = sub_train_df["forecast_kst_dtm"].dt.to_period("M")

    oof_preds_target = np.zeros(len(sub_train_df))
    cap = config["capacity_kwh"][target]

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_target, y_target, groups)):
        X_tr, y_tr = X_target.iloc[train_idx], y_target.iloc[train_idx]
        X_val, y_val = X_target.iloc[val_idx], y_target.iloc[val_idx]

        if model_type == "LightGBM":
            model = LGBMRegressor(**config["model_params"])
        elif model_type == "XGBoost":
            model = XGBRegressor(**config["model_params"])
        elif model_type == "CatBoost":
            model = CatBoostRegressor(**config["model_params"])
        elif model_type == "RandomForest":
            model = RandomForestRegressor(**config["model_params"])
        else:
            raise ValueError(f"지원하지 않는 model_type입니다: {model_type}")

        model.fit(X_tr, y_tr)
        
        # Fold별 모델 저장
        joblib.dump(model, save_dir / f"model_{target}_fold{fold}.pkl")

        # Validation 예측 및 상한선 클리핑
        val_pred = model.predict(X_val)
        oof_preds_target[val_idx] = np.clip(val_pred, 0, cap)

    oof_pred_df.loc[train_mask, target] = oof_preds_target

# 5. OOF 기반 평가 산식 적용 (후처리 적용 전)
answer_df = train_df[config["targets"]].copy()
raw_score, raw_nmae, raw_ficr = calculate_metric(answer_df, oof_pred_df)

# 6. FICR 최적화 후처리 파라미터 탐색 및 적용
print("\n⚙️ [Post-Processing] FICR 극대화 최적화 파라미터 탐색 중...")
post_params = optimize_postprocessing(answer_df, oof_pred_df)
joblib.dump(post_params, save_dir / "post_params.pkl")

oof_pred_post_df = apply_postprocessing(oof_pred_df, post_params)
total_score, one_minus_nmae, ficr = calculate_metric(answer_df, oof_pred_post_df)

print("=" * 50)
print(f"📊 [{config['version']}] Post-Processed OOF Result")
print(f" - Total Score : {raw_score:.4f} -> {total_score:.4f} (▲ {total_score - raw_score:+.4f})")
print(f" - 1 - NMAE    : {raw_nmae:.4f} -> {one_minus_nmae:.4f}")
print(f" - FICR        : {raw_ficr:.4f} -> {ficr:.4f}")
print("=" * 50)

# 6. 엑셀 로깅
log_experiment(
    config=config,
    total_score=total_score,
    one_minus_nmae=one_minus_nmae,
    ficr=ficr,
    features_summary=config.get("features_summary", ""),
    notes=f"{n_splits}-Fold GroupKFold(월별) OOF / " + config.get("notes", "")
)

print(f"✅ 학습 완료! 모델 및 설정 파일이 '{save_dir}'에 저장되었습니다.")