import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

from src.utils import seed_everything, calculate_metric
from src.features import process_calendar_features, add_wind_features
from src.logger import log_experiment

# 1. Config 로드
with open("./configs/config_v2.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

seed_everything(config["seed"])
save_dir = Path(config["data_paths"]["save_model_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

# 2. 데이터 로드
train_dir = Path(config["data_paths"]["train_dir"])
train_labels = pd.read_csv(train_dir / "train_labels.csv", encoding="utf-8-sig")
ldaps_train = pd.read_csv(train_dir / "ldaps_train.csv", encoding="utf-8-sig")
gfs_train = pd.read_csv(train_dir / "gfs_train.csv", encoding="utf-8-sig")

def aggregate_weather(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    return agg.reset_index()

train_weather = aggregate_weather(ldaps_train, "ldaps").merge(
    aggregate_weather(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
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

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
joblib.dump(imputer, save_dir / "imputer.pkl")

# 4. 모델 학습 및 예측
pred_train_df = pd.DataFrame(index=train_df.index)
model_type = config.get("model_type", "RandomForest")

for target in config["targets"]:
    train_mask = train_df[target].notna()
    y_train = train_df.loc[train_mask, target]
    X_tr = X_train_imp.loc[train_mask]
    
# config 설정에 따라 Target별 독립 모델 인스턴스 생성
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
    
    model.fit(X_tr, y_train)
    
    joblib.dump(model, save_dir / f"model_{target}.pkl")
    
    pred = model.predict(X_train_imp)
    cap = config["capacity_kwh"][target]
    pred_train_df[target] = np.clip(pred, 0, cap)

# 5. 데이콘 공식 평가 산식 적용
answer_df = train_df[config["targets"]].copy()
total_score, one_minus_nmae, ficr = calculate_metric(answer_df, pred_train_df)

print("=" * 50)
print(f"📊 [{config['version']}] Train Evaluation Metric Result")
print(f" - Total Score     : {total_score:.4f}")
print(f" - 1 - NMAE        : {one_minus_nmae:.4f}")
print(f" - FICR            : {ficr:.4f}")
print("=" * 50)

# 6. 엑셀에 자동으로 실험 결과 로깅 (추가된 부분)
log_experiment(
    config=config,
    total_score=total_score,
    one_minus_nmae=one_minus_nmae,
    ficr=ficr,
    features_summary=config.get("features_summary", ""),
    notes=config.get("notes", "")
)

print(f"✅ 학습 완료! 모델 및 객체가 '{save_dir}'에 성공적으로 저장되었습니다.")