import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.features import process_calendar_features, add_wind_features
from src.postprocessing import apply_postprocessing

# 1. Config 로드
config_path = "./configs/config_v5.yaml"  
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_dir = Path(config["data_paths"]["save_model_dir"])
test_dir = Path(config["data_paths"]["test_dir"])
sub_dir = Path(config["data_paths"]["submission_dir"])
sub_dir.mkdir(parents=True, exist_ok=True)

# 2. Test 및 피처 명세 데이터 로드
print("🔄 [Data Load] Test 데이터 및 피처 명세 로드 중...")
sample_sub = pd.read_csv("./sample_submission.csv", encoding="utf-8-sig")
ldaps_test = pd.read_csv(test_dir / "ldaps_test.csv", encoding="utf-8-sig")
gfs_test = pd.read_csv(test_dir / "gfs_test.csv", encoding="utf-8-sig")

# 학습 단계에서 저장된 피처 명세 불러오기
feature_cols_path = model_dir / "feature_cols.pkl"
if not feature_cols_path.exists():
    raise FileNotFoundError(
        f"피처 명세 파일({feature_cols_path})을 찾을 수 없습니다. 'main_train.py'를 먼저 실행해 주세요."
    )
feature_cols = joblib.load(feature_cols_path)

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

test_weather = process_weather_data(ldaps_test, "ldaps").merge(
    process_weather_data(gfs_test, "gfs"), on="forecast_kst_dtm", how="inner"
)

sample_sub["forecast_kst_dtm"] = pd.to_datetime(sample_sub["forecast_kst_dtm"])
test_df = sample_sub[["forecast_id", "forecast_kst_dtm"]].merge(
    test_weather, on="forecast_kst_dtm", how="left"
)

# 3. 피처 생성 및 학습 피처 목록에 맞게 재정렬
cal_feat = process_calendar_features(test_df["forecast_kst_dtm"])
X_test_raw = pd.concat(
    [cal_feat, test_df.drop(columns=["forecast_id", "forecast_kst_dtm"])],
    axis=1
)
X_test_raw = add_wind_features(X_test_raw)

# 무한대(inf) 정제
X_test_imp = X_test_raw.replace([np.inf, -np.inf], np.nan)

# 학습 시 피처 순서 및 컬럼 구성과 동일하게 강제 재정렬 (부족한 컬럼은 NaN 채움)
X_test_imp = X_test_imp.reindex(columns=feature_cols)

# 4. 추론 및 K-Fold 모델 앙상블
submission = sample_sub[["forecast_id", "forecast_kst_dtm"]].copy()
n_splits = config.get("n_splits", 5)

for target in config["targets"]:
    print(f"🔮 [Inference: {target}] {n_splits}-Fold 앙상블 추론 진행 중...")
    target_preds = np.zeros(len(X_test_imp))
    cap = config["capacity_kwh"][target]
    
    for fold in range(n_splits):
        model_path = model_dir / f"model_{target}_fold{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일({model_path})을 찾을 수 없습니다.")
            
        model = joblib.load(model_path)
        fold_pred = model.predict(X_test_imp)
        target_preds += fold_pred / n_splits
    
    # 상한선(그룹별 설비용량) 및 하한선(0) 클리핑 적용
    submission[target] = np.clip(target_preds, 0, cap)

# 5. 최적화 후처리 적용
post_params_path = model_dir / "post_params.pkl"
if post_params_path.exists():
    print("⚙️ [Post-Processing] 최적화 후처리 파라미터 적용 중...")
    post_params = joblib.load(post_params_path)
    submission = apply_postprocessing(submission, post_params)

# 6. 제출 파일 저장
submission["forecast_kst_dtm"] = submission["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
out_path = sub_dir / f"submit_{config['version']}.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"\n🚀 추론 완료! 제출 파일이 저장되었습니다: {out_path}")