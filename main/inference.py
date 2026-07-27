"""
main/inference_v11.py — 기존 main/inference.py 교체본

⚠️ 이거 같이 안 바꾸면 터진다:
   run_cv 가 저장하는 모델 파일명이 'model_{target}_fold{i}.pkl' 이 아니라
   'model_{target}_val2022.pkl' 처럼 fold 이름 기반으로 바뀐다 (LOYO 라 fold 개수도
   연도 수만큼). 기존 inference.py 는 range(n_splits) 로 fold0~4 를 찾으므로
   FileNotFoundError 가 난다. 여기서는 glob 으로 찾아서 두 방식 다 지원한다.

변경점
  1. config 를 CLI 인자로 (기존: config_v6.yaml 하드코딩 — train.py 는 v8 이었음)
  2. process_weather_data 가 data_available_kst_dtm 을 살려서 내보냄 (train 과 동일해야 함)
  3. 모델 파일 glob 탐색 + 몇 개를 평균했는지 출력
  4. 후처리 전 raw 예측 저장 (앙상블 이중 후처리 방지 — 기존 로직 유지)
"""
import argparse
import json

import sys
from pathlib import Path

# main/ 에서 직접 실행해도 src 를 찾을 수 있게 프로젝트 루트를 경로에 추가한다.
#   python main/train.py ... 로 실행하면 sys.path[0] 이 main/ 이 되어 src 를 못 찾는다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features import build_full_feature_pipeline
from src.postprocessing import apply_postprocessing
from src.features_grid import attach as attach_grid, marker_enabled   # ← 추가
from src.features_anom import attach_test as attach_anom_test, marker_enabled as anom_enabled
from src.features_obs import attach_test as attach_obs_test, marker_enabled as obs_enabled


def process_weather_data(df, prefix):
    """train_v11.py 와 반드시 동일해야 한다."""
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"latitude", "longitude"}
    id_cols = {"forecast_kst_dtm", "grid_id", "data_available_kst_dtm"}
    value_cols = [c for c in df.columns if c not in (id_cols | drop_cols)]

    pivoted = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=value_cols)
    pivoted.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    agg = agg.reset_index()

    avail = (df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first().reset_index()
             if "data_available_kst_dtm" in df.columns else None)

    out = pivoted.merge(agg, on="forecast_kst_dtm", how="inner")
    if avail is not None:
        out = out.merge(avail, on="forecast_kst_dtm", how="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-post", action="store_true", help="후처리 없이 raw 예측만 저장")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    model_dir = Path(config["data_paths"]["save_model_dir"])
    test_dir = Path(config["data_paths"]["test_dir"])
    sub_dir = Path(config["data_paths"]["submission_dir"]); sub_dir.mkdir(parents=True, exist_ok=True)
    targets = config["targets"]

    print("🔄 Test 데이터 로드 중...")
    # sample_submission.csv 위치가 프로젝트마다 달라서 후보를 순회한다
    cands = [Path("./sample_submission.csv"), test_dir / "sample_submission.csv",
             Path(config["data_paths"].get("train_dir", "./train")).parent / "sample_submission.csv",
             Path("./data/sample_submission.csv")]
    sub_path = next((c for c in cands if c.exists()), None)
    if sub_path is None:
        found = list(Path(".").rglob("sample_submission.csv"))
        if found:
            sub_path = found[0]
        else:
            raise FileNotFoundError(
                "sample_submission.csv 를 찾을 수 없다. 찾아본 곳: "
                + ", ".join(str(c) for c in cands))
    print(f"   제출 양식: {sub_path}")
    sample = pd.read_csv(sub_path, encoding="utf-8-sig")
    ldaps = pd.read_csv(test_dir / "ldaps_test.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(test_dir / "gfs_test.csv", encoding="utf-8-sig")

    feature_cols = joblib.load(model_dir / "feature_cols.pkl")

    w = process_weather_data(ldaps, "ldaps").merge(
        process_weather_data(gfs, "gfs"), on="forecast_kst_dtm",
        how="inner", suffixes=("", "_gfsdup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_gfsdup")])

    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    test_df = sample[["forecast_id", "forecast_kst_dtm"]].merge(w, on="forecast_kst_dtm", how="left")

    print("🚀 피처 생성 중...")
    X = build_full_feature_pipeline(test_df).replace([np.inf, -np.inf], np.nan)
    if marker_enabled(model_dir):                             # ← 추가 3줄
        print("🗺️  격자 산포 피처 활성 (학습 시 사용됨)")
        X = attach_grid(X, ldaps, gfs)
    if anom_enabled(model_dir):                               # ← 추가 (step32)
        print("🌊 후행편차 피처 활성 — 학습 꼬리(anom_state.csv)를 이어붙여 계산")
        X = attach_anom_test(X, model_dir)
    if obs_enabled(model_dir):                                # ← 추가 (step33)
        print("📡 관측 피처 활성 — ASOS csv + 학습 꼬리(obs_state.csv)로 계산")
        X = attach_obs_test(X, model_dir, "./data/external/asos_hourly.csv")
    X = X.reindex(columns=feature_cols)
    missing = X.columns[X.isna().all()].tolist()
    if missing:
        print(f"   ⚠️ 학습에는 있었는데 테스트에서 전부 NaN 인 피처 {len(missing)}개: {missing[:8]}...")

    jg_path = model_dir / "joint_groups.json"
    joint_groups = json.load(open(jg_path)) if jg_path.exists() else []
    if joint_groups:
        print(f"🔗 통합 모델 그룹: {', '.join(joint_groups)}")

    scale_path = model_dir / "target_scale.json"
    scale = json.load(open(scale_path)) if scale_path.exists() else None
    if scale:
        print("🧹 정제 타깃 모델 감지 — 가용률 보정 예측에 k×평균가용률을 곱해 실제 kWh 로 환산: "
              + ", ".join(f"{k}={v:.4f}" for k, v in scale.items()))

    from src.validation import GROUP_SPEC, CREF

    def joint_X(base_X, target):
        """통합 모델 입력: 피처 + 그룹 one-hot + 정적 스펙"""
        sp = GROUP_SPEC[target]
        d = base_X.astype(np.float32).copy()
        for k in range(3):
            d[f"grp_{k}"] = np.float32(1.0 if sp["gid"] == k else 0.0)
        d["grp_is_vestas"] = np.float32(sp["is_vestas"])
        d["grp_n_turb"] = np.float32(sp["n_turb"])
        d["grp_rotor_d"] = np.float32(sp["rotor_d"])
        return d

    submission = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    for target in targets:
        if target in joint_groups:
            jp = sorted(model_dir.glob("model_joint_*.pkl"))
            if not jp:
                raise FileNotFoundError(f"{model_dir} 에 model_joint_*.pkl 이 없다.")
            cap = config["capacity_kwh"][target]
            Xj = joint_X(X, target)
            preds = np.zeros(len(X))
            for p in jp:
                fold = p.stem.split("_")[-1]
                fc = model_dir / f"featcols_joint_{fold}.pkl"
                Xi = Xj[joblib.load(fc)] if fc.exists() else Xj
                preds += np.clip(joblib.load(p).predict(Xi), 0, 1.15 * CREF) / len(jp)
            submission[target] = np.clip(preds * (scale[target] if scale else 1.0), 0, cap)
            print(f"🔗 {target}: 통합 모델 {len(jp)}개 평균")
            continue
        paths = sorted(model_dir.glob(f"model_{target}_*.pkl"))
        if not paths:
            raise FileNotFoundError(f"{model_dir} 에 model_{target}_*.pkl 이 없습니다. 먼저 학습하세요.")
        cap = config["capacity_kwh"][target]
        ub = 1.15 * cap if scale else cap                # 가용률 보정 타깃은 cap 을 넘을 수 있다
        preds = np.zeros(len(X))
        for p in paths:
            fold = p.stem.split("_")[-1]
            fc = model_dir / f"featcols_{target}_{fold}.pkl"
            Xi = X[joblib.load(fc)] if fc.exists() else X      # 가지치기 모델은 폴드별 피처
            preds += np.clip(joblib.load(p).predict(Xi), 0, ub) / len(paths)
        if scale:
            preds = preds * scale[target]
        submission[target] = np.clip(preds, 0, cap)
        nfe = [len(joblib.load(model_dir / f"featcols_{target}_{p.stem.split('_')[-1]}.pkl"))
               for p in paths if (model_dir / f"featcols_{target}_{p.stem.split('_')[-1]}.pkl").exists()]
        extra = f", 피처 {sorted(set(nfe))}개" if nfe else ""
        print(f"🔮 {target}: fold 모델 {len(paths)}개 평균  "
              f"({', '.join(p.stem.split('_')[-1] for p in paths)}{extra})")

    raw_path = model_dir / "raw_test_preds.csv"
    submission.to_csv(raw_path, index=False)
    print(f"📦 후처리 전 raw 예측 저장: {raw_path}  (앙상블은 반드시 이 파일을 쓸 것)")

    if not args.no_post:
        pp = model_dir / "post_params.pkl"
        if pp.exists():
            submission = apply_postprocessing(submission, joblib.load(pp))
            print("⚙️ 후처리 적용 완료")
        else:
            print("⚠️ post_params.pkl 없음 — raw 유지")

    submission["forecast_kst_dtm"] = submission["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # 후처리 없는 제출은 같은 버전의 '변형' 이다. 새 버전이 아니라 접미사로 구분한다
    # (덮어쓰면 v14 본편 제출물이 사라져서 나중에 둘을 비교할 수 없다).
    suffix = "_no_post" if args.no_post else ""
    out = sub_dir / f"submit_{config.get('version','v11')}{suffix}.csv"
    submission.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n🚀 제출 파일 저장: {out}")


if __name__ == "__main__":
    main()