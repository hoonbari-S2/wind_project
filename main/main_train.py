import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
# from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils import seed_everything, calculate_metric
from src.features import process_calendar_features, add_wind_features
from src.logger import log_experiment

# 1. Config 로드
with open("./configs/config_v5.yaml", "r", encoding="utf-8") as f:
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

train_weather = process_weather_data(ldaps_train, "ldaps").merge(
    process_weather_data(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
)

train_base = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
train_base["forecast_kst_dtm"] = pd.to_datetime(train_base["forecast_kst_dtm"])
train_df = train_base.merge(train_weather, on="forecast_kst_dtm", how="left")

# 3. 피처 생성 및 Impute
cal_feat = process_calendar_features(train_df["forecast_kst_dtm"])
X_train_raw = pd.concat(
    [cal_feat, train_df.drop(columns=["forecast_kst_dtm", *config["targets"]])],
    axis=1
)
X_train_raw = add_wind_features(X_train_raw)

# lgbm 사용으로 imputer 제외, inf값만 np.nan으로 변환(멱법칙 계산 시 분모가 0이 될수 있기 때문)
# imputer = SimpleImputer(strategy="median")
# X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
# joblib.dump(imputer, save_dir / "imputer.pkl")

X_train_imp = X_train_raw.replace([np.inf, -np.inf], np.nan)

# 4. K-Fold Cross Validation 및 OOF 추론
n_splits = config.get("n_splits", 5)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=config["seed"])

oof_pred_df = pd.DataFrame(index=train_df.index, columns=config["targets"], dtype=float)
model_type = config.get("model_type", "RandomForest")

for target in config["targets"]:
    print(f"\n🌀 [Target: {target}] {n_splits}-Fold Cross Validation 진행 중...")
    train_mask = train_df[target].notna()
    y_target = train_df.loc[train_mask, target].values
    X_target = X_train_imp.loc[train_mask].values

    oof_preds_target = np.zeros(len(X_target))
    cap = config["capacity_kwh"][target]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_target, y_target)):
        X_tr, y_tr = X_target[train_idx], y_target[train_idx]
        X_val, y_val = X_target[val_idx], y_target[val_idx]

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

        # Validation 예측
        val_pred = model.predict(X_val)
        oof_preds_target[val_idx] = np.clip(val_pred, 0, cap)

    oof_pred_df.loc[train_mask, target] = oof_preds_target

# 5. OOF 기반 평가 산식 적용
answer_df = train_df[config["targets"]].copy()
total_score, one_minus_nmae, ficr = calculate_metric(answer_df, oof_pred_df)

print("=" * 50)
print(f"📊 [{config['version']}] OOF Evaluation Metric Result")
print(f" - Total Score     : {total_score:.4f}")
print(f" - 1 - NMAE        : {one_minus_nmae:.4f}")
print(f" - FICR            : {ficr:.4f}")
print("=" * 50)

# 6. 엑셀 로깅
log_experiment(
    config=config,
    total_score=total_score,
    one_minus_nmae=one_minus_nmae,
    ficr=ficr,
    features_summary=config.get("features_summary", ""),
    notes=f"{n_splits}-Fold OOF 평가 / " + config.get("notes", "")
)

print(f"✅ 학습 완료! 모델이 '{save_dir}'에 저장되었습니다.")