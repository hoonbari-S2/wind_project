import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.features import process_calendar_features, add_wind_features

# 1. Config 로드
with open("./configs/config_v5.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_dir = Path(config["data_paths"]["save_model_dir"])
test_dir = Path(config["data_paths"]["test_dir"])
sub_dir = Path(config["data_paths"]["submission_dir"])
sub_dir.mkdir(parents=True, exist_ok=True)

# 2. Test 데이터 로드
sample_sub = pd.read_csv("./sample_submission.csv", encoding="utf-8-sig")
ldaps_test = pd.read_csv(test_dir / "ldaps_test.csv", encoding="utf-8-sig")
gfs_test = pd.read_csv(test_dir / "gfs_test.csv", encoding="utf-8-sig")
# imputer = joblib.load(model_dir / "imputer.pkl")

def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", "grid_id", *drop_cols}]
    
    pivoted = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=value_cols)
    pivoted.columns = [f"{prefix}_g{col[1]}_{col[0]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    agg_mean = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg_mean.columns = [f"{prefix}_mean_{c}" for c in agg_mean.columns]
    agg_mean = agg_mean.reset_index()
    
    return pivoted.merge(agg_mean, on="forecast_kst_dtm", how="inner")

test_weather = process_weather_data(ldaps_test, "ldaps").merge(
    process_weather_data(gfs_test, "gfs"), on="forecast_kst_dtm", how="inner"
)

sample_sub["forecast_kst_dtm"] = pd.to_datetime(sample_sub["forecast_kst_dtm"])
test_df = sample_sub[["forecast_id", "forecast_kst_dtm"]].merge(
    test_weather, on="forecast_kst_dtm", how="left"
)

# 3. 피처 변환 및 Impute
cal_feat = process_calendar_features(test_df["forecast_kst_dtm"])
X_test_raw = pd.concat(
    [cal_feat, test_df.drop(columns=["forecast_id", "forecast_kst_dtm"])],
    axis=1
)
X_test_raw = add_wind_features(X_test_raw)
# X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

X_test_imp = X_test_raw.replace([np.inf, -np.inf], np.nan)

# 4. 추론 및 K-Fold 모델 앙상블
submission = sample_sub[["forecast_id", "forecast_kst_dtm"]].copy()
n_splits = config.get("n_splits", 5)

for target in config["targets"]:
    target_preds = np.zeros(len(X_test_imp))
    cap = config["capacity_kwh"][target]
    
    for fold in range(n_splits):
        model_path = model_dir / f"model_{target}_fold{fold}.pkl"
        model = joblib.load(model_path)
        fold_pred = model.predict(X_test_imp)
        target_preds += fold_pred / n_splits
    
    submission[target] = np.clip(target_preds, 0, cap)

# 5. 제출 파일 저장
submission["forecast_kst_dtm"] = submission["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
out_path = sub_dir / f"submit_{config['version']}.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"🚀 추론 완료! 제출 파일이 저장되었습니다: {out_path}")