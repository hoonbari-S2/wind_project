"""
==============================================================================
바뀐 것
  1. config 를 CLI 인자로 받는다.  (기존: train=config_v8, inference=config_v6 하드코딩)
       python main/train_v11.py --config configs/config_v6.yaml
  2. Early stopping 을 검증폴드가 아니라 '학습셋 내부의 예보블록'에서 한다.
  3. 검증을 LOYO(연 단위) 로 바꾼다. 테스트가 2025년 통째이므로 이게 같은 구조.
  4. 점수를 연도별로 쪼개서 '오차막대'와 함께 보고한다.
     -> 이 대회에서 관측되는 연도간 변동은 대략 ±0.01 규모다. 그보다 작은
        차이를 보고 제출을 쓰면 노이즈를 쫓는 것이다.
  5. 후처리 개선폭을 '중첩(nested)' 으로 측정한다. 같은 OOF에 fit 하고
     같은 OOF로 평가하면 개선폭이 부풀려진다.
==============================================================================
"""
import argparse, time, json

import sys
from pathlib import Path

# main/ 에서 직접 실행해도 src 를 찾을 수 있게 프로젝트 루트를 경로에 추가한다.
#   python main/train.py ... 로 실행하면 sys.path[0] 이 main/ 이 되어 src 를 못 찾는다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.utils import seed_everything, CAPACITY_KWH
from src.features import build_full_feature_pipeline
from src.logger import log_experiment
from src.postprocessing import optimize_postprocessing, apply_postprocessing
from src.validation import (run_cv, total_score, score_by_year, add_time_keys,
                            nested_postprocess_score, is_difference_real)


def process_weather_data(df, prefix):
    """[변경] data_available_kst_dtm 을 살려서 내보낸다 (lead time / 블록 경계용)."""
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
    ap.add_argument("--scheme", default="loyo",
                    choices=["loyo", "holdout2024", "block_month", "month_group"])
    ap.add_argument("--es-mode", default="refit", choices=["refit", "inner", "fixed"])
    ap.add_argument("--post-mode", default="piecewise", choices=["piecewise", "linear"])
    ap.add_argument("--compare-baseline", default=None,
                    help="이전 버전 oof_preds.csv 경로. 주면 연도별 짝비교로 유의성까지 판정")
    ap.add_argument("--clean-target", action="store_true",
                    help="SCADA 가동률로 정제한 cf 를 학습 타깃으로 사용 (step2 에서 +0.0126 확인)")
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--audit", action="store_true", help="피처 인과성 감사 (느림)")
    args = ap.parse_args()

    t0 = time.time()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(config["seed"])
    save_dir = Path(config["data_paths"]["save_model_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    targets = config["targets"]

    # ---------------- 데이터 ----------------
    tr_dir = Path(config["data_paths"]["train_dir"])
    labels = pd.read_csv(tr_dir / "train_labels.csv", encoding="utf-8-sig")
    ldaps = pd.read_csv(tr_dir / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(tr_dir / "gfs_train.csv", encoding="utf-8-sig")

    print("🔄 기상 데이터 피벗 중...")
    w = process_weather_data(ldaps, "ldaps").merge(
        process_weather_data(gfs, "gfs"), on="forecast_kst_dtm",
        how="inner", suffixes=("", "_gfsdup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_gfsdup")])

    base = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    base["forecast_kst_dtm"] = pd.to_datetime(base["forecast_kst_dtm"])
    df = base.merge(w, on="forecast_kst_dtm", how="left")

    df_raw = df.copy()                       # 인과성 감사용 (피처 생성 전)
    print("🚀 피처 생성 중...")
    df = build_full_feature_pipeline(df).replace([np.inf, -np.inf], np.nan)
    df = add_time_keys(df)

    if args.audit:
        from src.causality import assert_block_structure, audit_causality
        assert_block_structure(df_raw)
        ok, rep = audit_causality(df_raw, build_full_feature_pipeline, n_blocks=5)
        assert ok, f"규칙 3항 위반 피처: {rep.feature.tolist()}"

    # ---------------- 정제 타깃 (선택) ----------------
    # 원본 라벨은 answer_raw 로 보존한다. 채점은 반드시 원본 kWh 기준으로 해야 한다.
    answer = df[targets].copy()
    scale = None
    if args.clean_target:
        ct = pd.read_csv(Path(args.scada_dir) / "clean_target.csv", encoding="utf-8-sig")
        ct["forecast_kst_dtm"] = pd.to_datetime(ct.pop("kst_dtm"))
        df = df.merge(ct, on="forecast_kst_dtm", how="left")
        # 타깃을 cf(0~1) 가 아니라 'kWh 스케일' 로 둔다.
        #   reg:tweedie 의 편차는 스케일 불변이 아니다 (p=1.5 -> sqrt(c) 로 스케일).
        #   cf 로 바꾸면 손실이 sqrt(21600)≈147배 작아지는데 reg_lambda/min_child_weight
        #   는 그대로라 사실상 147배 과규제가 된다. 조건 A 와 같은 단위로 맞춘다.
        #   target = label * capacity / (k * 가용용량)  = "전 터빈 정상이었다면 나왔을 kWh"
        scale = {}
        for g in targets:
            if f"{g}_cf" not in df:
                raise KeyError(f"{g}_cf 없음. scripts/step1_scada_check.py 를 먼저 실행할 것.")
            cap = CAPACITY_KWH[g]
            tgt = df[f"{g}_cf"].to_numpy(float) * cap
            lab = answer[g].to_numpy(float)
            ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * cap)
            # label / target = k * 가용률.  추론 때 이 값을 곱해 실제 kWh 로 되돌린다.
            scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
            df[g] = np.clip(tgt, 0, 1.15 * cap)
        json.dump(scale, open(save_dir / "target_scale.json", "w"), indent=2)
        print("🧹 정제 타깃 사용 (가용률 보정 kWh). 추론 환산 배율(k×평균가용률): "
              + ", ".join(f"{g}={scale[g]:.4f}" for g in targets))

    exclude = (["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm",
                "year_month", "_year", "_fday"] + targets
               + [f"{g}_cf" for g in targets] + [f"{g}_avail_frac" for g in targets])
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    joblib.dump(feature_cols, save_dir / "feature_cols.pkl")
    print(f"📦 피처 {len(feature_cols)}개 / 행 {len(df)}개")
    print(f"   연도별 라벨 수: "
          + ", ".join(f"{t}={df.groupby('_year')[t].count().to_dict()}" for t in targets[:1]))

    # ---------------- 누설 없는 CV ----------------
    oof, fold_df, _ = run_cv(df, feature_cols, targets,
                             config.get("model_type", "XGBoost"),
                             config.get("model_params", {}),
                             scheme=args.scheme, n_splits=config.get("n_splits", 5),
                             es_rounds=config.get("early_stopping_rounds", 50),
                             es_mode=args.es_mode, seed=config["seed"],
                             save_dir=str(save_dir))
    if scale is not None:                               # cf -> kWh 로 되돌린다
        for g in targets:
            oof[g] = np.clip(oof[g] * scale[g], 0, CAPACITY_KWH[g])

    oof.to_csv(save_dir / "oof_preds.csv")
    fold_df.to_csv(save_dir / "fold_scores.csv", index=False)

    raw = total_score(answer, oof, targets)
    print("\n" + "=" * 74)
    print(f"📊 [{config.get('version','?')}] 누설 없는 OOF (scheme={args.scheme}, es={args.es_mode})")
    print(f"   Total {raw[0]:.4f}   1-NMAE {raw[1]:.4f}   FICR {raw[2]:.4f}")
    print("\n📐 [오차막대] 연도 하나가 곧 1회 시행이다. 이 표준편차보다 작은 차이는 노이즈.")
    _, _, spread = score_by_year(df, answer, oof, targets, label="raw OOF")

    # ---------------- 후처리 (중첩 평가) ----------------
    nested, _, nst = nested_postprocess_score(
        df, answer, oof, targets, optimize_postprocessing, apply_postprocessing,
        mode=args.post_mode, verbose=True)
    score_by_year(df, answer, nested, targets, label="후처리 후")

    if not np.isnan(spread) and (nst[0] - raw[0]) < spread:
        print(f"\n   ⚠️  후처리 개선폭({nst[0]-raw[0]:+.4f})이 연도간 표준편차({spread:.4f})보다 작다.")
    else:
        print(f"\n   ✅ 후처리 개선폭({nst[0]-raw[0]:+.4f})이 연도간 변동을 넘어선다.")

    # 제출용 최종 파라미터는 OOF 전체로 fit (중첩 수치는 '얼마나 믿을지'를 알려주는 용도)
    post_params = optimize_postprocessing(answer, oof, mode=args.post_mode, verbose=False)
    joblib.dump(post_params, save_dir / "post_params.pkl")
    json.dump({"scheme": args.scheme, "es_mode": args.es_mode, "post_mode": args.post_mode,
               "clean_target": bool(args.clean_target), "target_scale": scale,
               "raw_oof": raw[0], "nested_post": nst[0], "year_spread": spread},
              open(save_dir / "validation_report.json", "w"), indent=2, default=float)

    # ---------------- 이전 버전과 짝비교 ----------------
    if args.compare_baseline:
        prev = pd.read_csv(args.compare_baseline, index_col=0)
        prev = prev.reindex(index=oof.index, columns=targets)
        print(f"\n📐 [이전 버전과 연도별 짝비교] {args.compare_baseline}")
        is_difference_real(df, answer, nested, prev,
                           name_a=config.get("version", "new"), name_b="baseline")

    print("=" * 74)
    # features_summary / notes 는 yaml 값을 우선 사용한다 (logger 내부에서 처리)
    log_experiment(
        config=config, total_score=nst[0], one_minus_nmae=nst[1], ficr=nst[2],
        execution_time_sec=time.time() - t0,
        validation=args.scheme,
        target_kind="clean_cf" if args.clean_target else "raw_label",
        es_mode=args.es_mode, post_mode=args.post_mode,
        raw_oof=raw[0], year_spread=spread, n_features=len(feature_cols),
        features_summary=f"{len(feature_cols)} feats"
                         + (" / clean_target(cf)" if args.clean_target else ""),
        notes=f"raw {raw[0]:.4f} -> nested post {nst[0]:.4f}")


if __name__ == "__main__":
    main()