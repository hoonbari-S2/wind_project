import os
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from src.utils import calculate_metric, TARGET_COLS
from src.logger import log_experiment
from src.postprocessing import optimize_postprocessing, apply_postprocessing
from src.features import build_full_feature_pipeline


def process_weather_data(df, prefix):
    """train.py의 기상 데이터 피벗 로직 동일 적용"""
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


def load_and_preprocess_train_data(data_paths):
    """학습 데이터 및 기상 피처 전처리 실행"""
    train_dir = Path(data_paths["train_dir"])
    train_labels = pd.read_csv(train_dir / "train_labels.csv", encoding="utf-8-sig")
    ldaps_train = pd.read_csv(train_dir / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(train_dir / "gfs_train.csv", encoding="utf-8-sig")

    train_weather = process_weather_data(ldaps_train, "ldaps").merge(
        process_weather_data(gfs_train, "gfs"), on="forecast_kst_dtm", how="inner"
    )

    train_base = train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    train_base["forecast_kst_dtm"] = pd.to_datetime(train_base["forecast_kst_dtm"])
    df_full = train_base.merge(train_weather, on="forecast_kst_dtm", how="left")

    df_processed = build_full_feature_pipeline(df_full)
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    return df_processed


def reconstruct_oof_from_pkl(v, config, df_processed):
    """oof_preds.csv가 없을 경우 저장된 .pkl 모델들로 OOF를 복원합니다."""
    save_dir = Path(f"./saved_models/{v}")
    oof_path = save_dir / "oof_preds.csv"

    if oof_path.exists():
        print(f"  ├─ [{v}] 기존 'oof_preds.csv' 발견! 바로 로드합니다.")
        return pd.read_csv(oof_path, index_col=0)

    print(f"  ├─ [{v}] 'oof_preds.csv' 부재 -> 저장된 .pkl 모델들로 OOF 자동 복원 중...")

    feature_cols = joblib.load(save_dir / "feature_cols.pkl")
    targets = config["targets"]
    n_splits = config.get("n_splits", 5)
    gkf = GroupKFold(n_splits=n_splits)

    oof_df = pd.DataFrame(index=df_processed.index, columns=targets, dtype=float)

    for target in targets:
        train_mask = df_processed[target].notna()
        sub_df = df_processed.loc[train_mask].copy()
        X_target = sub_df[feature_cols].reset_index(drop=True)
        y_target = sub_df[target].reset_index(drop=True)
        groups = pd.to_datetime(sub_df["forecast_kst_dtm"]).dt.to_period("M").reset_index(drop=True)

        oof_preds_target = np.zeros(len(sub_df))
        cap = config["capacity_kwh"][target]

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_target, y_target, groups)):
            model_path = save_dir / f"model_{target}_fold{fold}.pkl"
            model = joblib.load(model_path)

            X_val = X_target.iloc[val_idx]
            val_pred = model.predict(X_val)
            oof_preds_target[val_idx] = np.clip(val_pred, 0, cap)

        oof_df.loc[train_mask, target] = oof_preds_target

    oof_df.to_csv(oof_path)
    print(f"  └─ ✅ [{v}] OOF 복원 완료 및 '{oof_path}' 저장 성공!")
    return oof_df


def main():
    start_time = time.time()

    versions = ["v6", "v7", "v8"]
    oof_dict = {}
    raw_test_pred_dict = {}

    # v10 설정 구성
    with open("./configs/config_v6.yaml", "r", encoding="utf-8") as f:
        config_v10 = yaml.safe_load(f)

    config_v10["version"] = "v10"
    config_v10["model_type"] = "Ensemble(XGB+LGBM+CAT)"
    save_dir = Path("./saved_models/v10")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(config_v10["data_paths"]["train_dir"])
    train_labels = pd.read_csv(train_dir / "train_labels.csv", encoding="utf-8-sig")
    targets = config_v10["targets"]
    answer_df = train_labels[targets].copy()

    # OOF 복원이 필요한 버전이 하나라도 있을 때만 학습 데이터 로드
    needs_reconstruction = any(
        not Path(f"./saved_models/{v}/oof_preds.csv").exists() for v in versions
    )

    df_processed = None
    if needs_reconstruction:
        print("📦 [앙상블] OOF 복원을 위한 학습 데이터 전처리 실행 중...")
        df_processed = load_and_preprocess_train_data(config_v10["data_paths"])

    print("📦 [앙상블] v6, v7, v8 원본 OOF 및 원본 테스트 예측 데이터 준비 중...")
    for v in versions:
        with open(f"./configs/config_{v}.yaml", "r", encoding="utf-8") as f:
            v_config = yaml.safe_load(f)

        # 1. 원본 OOF 로드 또는 자동 복원
        oof_dict[v] = reconstruct_oof_from_pkl(v, v_config, df_processed)

        # 2. 이중 후처리 방지를 위해 후처리 전 원본(Raw) 테스트 예측 데이터 로드
        raw_test_path = Path(f"./saved_models/{v}/raw_test_preds.csv")
        if not raw_test_path.exists():
            raise FileNotFoundError(
                f"❌ '{raw_test_path}' 파일을 찾을 수 없습니다.\n"
                f"이중 후처리를 막기 위해서는 후처리 전의 원본 테스트 예측 결과가 필요합니다.\n"
                f"main/inference.py 수정 후 각 버전 환경에서 'python run.py inference'를 실행해 주세요."
            )
        raw_test_pred_dict[v] = pd.read_csv(raw_test_path)

    # OOF 기반 SLSQP 최적 결합 가중치 탐색
    def loss_func(weights):
        w1, w2, w3 = weights
        ens_oof = answer_df.copy()
        for col in targets:
            ens_oof[col] = (
                w1 * oof_dict["v6"][col] +
                w2 * oof_dict["v7"][col] +
                w3 * oof_dict["v8"][col]
            )
        total_score, _, _ = calculate_metric(answer_df, ens_oof)
        return -total_score

    init_weights = [1/3, 1/3, 1/3]
    bounds = [(0, 1), (0, 1), (0, 1)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

    print("\n⚙️ [최적화] OOF 기반 최적 모델 결합 가중치(w_XGB, w_LGBM, w_CAT) 탐색 중...")
    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_w = res.x / np.sum(res.x)

    print("🎯 최적 가중치 산출 완료:")
    print(f" - XGBoost (v6)  : {best_w[0]:.4f}")
    print(f" - LightGBM (v7) : {best_w[1]:.4f}")
    print(f" - CatBoost (v8) : {best_w[2]:.4f}")

    # 가중 평균 원본 OOF 생성
    ens_oof_raw = answer_df.copy()
    for col in targets:
        ens_oof_raw[col] = (
            best_w[0] * oof_dict["v6"][col] +
            best_w[1] * oof_dict["v7"][col] +
            best_w[2] * oof_dict["v8"][col]
        )

    raw_score, raw_1_nmae, raw_ficr = calculate_metric(answer_df, ens_oof_raw)

    print("\n⚙️ [후처리] 앙상블 OOF 기준 Nelder-Mead 후처리 재최적화 진행 중...")
    post_params = optimize_postprocessing(answer_df, ens_oof_raw)
    ens_oof_post = apply_postprocessing(ens_oof_raw, post_params)

    total_score, one_minus_nmae, ficr = calculate_metric(answer_df, ens_oof_post)
    execution_time_sec = time.time() - start_time

    print("=" * 60)
    print(f"📊 [v10 앙상블] 후처리 적용 최종 OOF 평가 결과")
    print(f" - 종합 점수 (Total Score) : {raw_score:.4f} -> {total_score:.4f} (▲ {total_score - raw_score:+.4f})")
    print(f" - 1 - NMAE              : {raw_1_nmae:.4f} -> {one_minus_nmae:.4f}")
    print(f" - FICR 정산 비율        : {raw_ficr:.4f} -> {ficr:.4f}")
    print("=" * 60)

    # 원본 테스트 예측치를 결합한 후 '단 1회만' Nelder-Mead 후처리 적용
    raw_sub_v10 = raw_test_pred_dict["v6"].copy()
    for col in targets:
        raw_sub_v10[col] = (
            best_w[0] * raw_test_pred_dict["v6"][col] +
            best_w[1] * raw_test_pred_dict["v7"][col] +
            best_w[2] * raw_test_pred_dict["v8"][col]
        )

    final_sub_v10 = apply_postprocessing(raw_sub_v10, post_params)
    sub_path = "./submissions/submit_v10.csv"
    final_sub_v10.to_csv(sub_path, index=False, encoding="utf-8-sig")
    print(f"📄 [제출 파일] v10 앙상블 최종 제출 파일 생성 완료: '{sub_path}'")

    log_experiment(
        config=config_v10,
        total_score=total_score,
        one_minus_nmae=one_minus_nmae,
        ficr=ficr,
        execution_time_sec=execution_time_sec,
        features_summary=f"v6/v7/v8 앙상블 (가중치: {best_w[0]:.2f}/{best_w[1]:.2f}/{best_w[2]:.2f})",
        notes=f"v10: 원본 예측치 가중 결합 후 Nelder-Mead 단일 후처리 적용 (+{total_score - raw_score:.4f})"
    )


if __name__ == "__main__":
    main()