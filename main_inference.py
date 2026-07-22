import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.features import process_calendar_features, add_wind_features

# 1. Config 로드
with open("./configs/config_v2.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_dir = Path(config["data_paths"]["save_model_dir"])
test_dir = Path(config["data_paths"]["test_dir"])
sub_dir = Path(config["data_paths"]["submission_dir"])
sub_dir.mkdir(parents=True, exist_ok=True)

# 2. Test 데이터 및 저장된 Imputer 로드
sample_sub = pd.read_csv("./sample_submission.csv", encoding="utf-8-sig")
ldaps_test = pd.read_csv(test_dir / "ldaps_test.csv", encoding="utf-8-sig")
gfs_test = pd.read_csv(test_dir / "gfs_test.csv", encoding="utf-8-sig")
imputer = joblib.load(model_dir / "imputer.pkl")

def aggregate_weather(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    return agg.reset_index()

test_weather = aggregate_weather(ldaps_test, "ldaps").merge(
    aggregate_weather(gfs_test, "gfs"), on="forecast_kst_dtm", how="inner"
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
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

# 4. 추론 및 Clipping
submission = sample_sub[["forecast_id", "forecast_kst_dtm"]].copy()

for target in config["targets"]:
    model = joblib.load(model_dir / f"model_{target}.pkl")
    pred = model.predict(X_test_imp)
    
    # 0 ~ 최대 용량 범위로 제한
    cap = config["capacity_kwh"][target]
    pred = np.clip(pred, 0, cap)
    submission[target] = pred

# 5. 제출 파일 저장
submission["forecast_kst_dtm"] = submission["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
out_path = sub_dir / f"submit_{config['version']}.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"🚀 추론 완료! 제출 파일이 저장되었습니다: {out_path}")