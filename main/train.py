"""
main/train_v11.py — 기존 main/train.py 교체본
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
from src.validation import (run_cv, run_cv_joint, total_score, group_score, score_by_year, add_time_keys,
                            nested_postprocess_score, is_difference_real)
from src.features import build_full_feature_pipeline
from src.features_grid import attach as attach_grid, save_marker    # ← 추가
from src.features_anom import attach as attach_anom, save_state as save_anom_state
from src.features_obs import attach as attach_obs, save_state as save_obs_state


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
    ap.add_argument("--top-k", type=int, default=None,
                    help="폴드별 상위 K개 피처만 사용 (step4 결과: 200~400 구간이 안정적)")
    ap.add_argument("--joint-groups", default="",
                    help="통합(long-format) 모델 예측을 쓸 그룹. 예: kpx_group_1,kpx_group_3 "
                         "(step5b: G1 3/3 +, G3 2/2 +, G2 는 3/3 - 이므로 독립 유지)")
    ap.add_argument("--objective", default=None, choices=["tweedie", "mae"],
                    help="config 의 objective 를 덮어쓴다. mae -> reg:absoluteerror. "
                         "(step9: L-T 주효과 +0.0137, 연도 3/3 일관)")
    ap.add_argument("--sample-weight", default="none", choices=["none", "ficr"],
                    help="ficr -> 산식정합 가중. 채점행에 0.5 균일 + 0.5 발전량비례, "
                         "채점제외행(actual<10%%cap)은 --low-weight. "
                         "(step9: W-U 주효과 +0.0118, 연도 3/3 일관)")
    ap.add_argument("--low-weight", type=float, default=0.2,
                    help="채점 제외행 가중치. 0 으로 두면 임계값 근처 캘리브레이션이 무너진다")
    ap.add_argument("--obs-features", action="store_true",
                    help="ASOS 공공 관측 피처 10개 (블록 상수. §8.8 / 규칙 3 실측 조항 준수)")
    ap.add_argument("--obs-csv", default="./data/external/asos_hourly.csv")
    ap.add_argument("--anom-features", action="store_true",
                    help="후행 720h 이동평균 편차 7개 추가 (step31/§3.31 드리프트의 표현 대응)")
    ap.add_argument("--year-weight", default=None,
                    help="연도별 학습 표본 가중. 예: 2022:0.5,2023:0.75,2024:1.0 "
                         "(step31: 습도↑·기온↑·서풍↓ 단조 드리프트 → 최신 해 강조. 학습 데이터만 사용)")
    ap.add_argument("--grid-features", action="store_true",
                    help="격자 간 산포·공간경도 54개 추가 (step23: raw +0.0023, 3/3, G3 최대)")
    ap.add_argument("--baseline-oof", default=None,
                    help="기준선 oof_preds.csv. 주면 raw OOF 짝비교로 게이트 판정")
    ap.add_argument("--featcols-from", default=None,
                    help="폴드별 피처를 이 디렉토리의 featcols_*.pkl 로 고정 (top-K 재선택 안 함). "
                         "피처 추가 A/B 에서 선택 절차를 상수로 만들기 위한 것 — v20 실측상 "
                         "풀에 10컬럼만 더해도 top-200 의 25%%가 뒤바뀌어 재추첨을 재게 됨")
    ap.add_argument("--force-prefix", default="",
                    help="쉼표 구분 접두사. 해당 컬럼은 선택 결과에 무조건 포함. 예: obs_ "
                         "('정보가 없다' 와 'gain 선택에서 떨어졌다' 를 가르는 유일한 방법)")
    ap.add_argument("--wgrid", action="store_true",
                    help="A-4 격자 가중 (v21). 통합 경로에 폴드별·그룹별 NNLS 가중 풍속 3컬럼을 "
                         "추가한다. 독립 경로(G2)는 손대지 않으므로 무손상 대조군이 됨")
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

    df_raw = df.copy()
    print("🚀 피처 생성 중...")
    df = build_full_feature_pipeline(df).replace([np.inf, -np.inf], np.nan)
    if args.grid_features:                                    # ← 추가 3줄
        df = attach_grid(df, ldaps, gfs)
    save_marker(save_dir, args.grid_features)
    if args.anom_features:                                    # ← 추가 (step32/§3.31)
        df = attach_anom(df)
        save_anom_state(save_dir, df)
    if args.obs_features:                                     # ← 추가 (step33/§8.8)
        df = attach_obs(df, args.obs_csv)
        save_obs_state(save_dir, df)
    df = add_time_keys(df)

    if args.audit:
        from src.causality import assert_block_structure, audit_causality
        assert_block_structure(df_raw)
        ok, rep = audit_causality(df_raw, build_full_feature_pipeline, n_blocks=5)
        assert ok, f"규칙 3항 위반 피처: {rep.feature.tolist()}"

    # ---------------- 목적함수 오버라이드 (선택) ----------------
    model_params = dict(config.get("model_params", {}))
    if args.objective == "mae":
        model_params.pop("tweedie_variance_power", None)
        model_params["objective"] = "reg:absoluteerror"
        model_params["eval_metric"] = "mae"
        print("🎯 목적함수: reg:absoluteerror (NMAE 는 용량정규화 L1 이므로 그룹 안에서 "
              "MAE 최소화 = NMAE 최소화. 또한 조건부 중앙값 예측이라 FICR 밴드에 더 많이 들어간다)")
    elif args.objective == "tweedie":
        model_params["objective"] = "reg:tweedie"
        model_params.setdefault("tweedie_variance_power", 1.5)
        model_params["eval_metric"] = "tweedie-nloglik@1.5"

    # ---------------- 정제 타깃 (선택) ----------------
    # 원본 라벨은 answer_raw 로 보존한다. 채점은 반드시 원본 kWh 기준으로 해야 한다.
    answer = df[targets].copy()
    scale = None

    # ---------------- 산식정합 표본가중 (선택) ----------------
    # 가중은 반드시 '원본 라벨' 로 만든다. 정제 타깃으로 만들면 채점 마스크가 어긋난다.
    weight_col = None
    if args.sample_weight == "ficr":
        weight_col = "_w_"
        for g in targets:
            a = answer[g].to_numpy(float)
            cap = CAPACITY_KWH[g]
            w = np.ones(len(a), dtype=float)
            fin = np.isfinite(a)
            scored = fin & (a >= 0.10 * cap)
            if scored.sum() >= 100:
                # 산식 0.5*(1-NMAE) + 0.5*FICR 의 구조를 그대로 옮긴다.
                #   NMAE 항  : 채점행에 균일          -> 0.5
                #   FICR 항  : 채점행에 발전량 비례   -> 0.5 * a/mean(a_scored)
                #   채점 제외행은 점수에 안 들어가지만 0 으로 버리면 임계값 근처
                #   캘리브레이션이 무너져 채점행 예측까지 흔들린다 -> low_weight
                w[fin] = args.low_weight
                w[scored] = 0.5 + 0.5 * a[scored] / a[scored].mean()
                w[fin] = w[fin] / w[fin].mean()            # 평균 1 (학습률과 분리)
            df[f"_w_{g}"] = w
        print(f"⚖️  산식정합 가중 사용 (채점제외행 {args.low_weight}). "
              + ", ".join(f"{g}:채점 {int((np.isfinite(answer[g])&(answer[g]>=0.10*CAPACITY_KWH[g])).sum()):,}행"
                          for g in targets))
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

    # ---------------- 연도 최신 가중 (선택, step31/§3.31) ----------------
    # ficr 가중과 곱으로 결합 가능. 반드시 평균 1 로 재정규화 (학습률과 분리).
    if args.year_weight:
        yw = {int(k): float(v) for k, v in (p_.split(":") for p_ in args.year_weight.split(","))}
        wy = np.array([yw.get(int(y), 1.0) for y in df["_year"].to_numpy()], dtype=float)
        weight_col = weight_col or "_w_"
        for g in targets:
            base_w = df[f"_w_{g}"].to_numpy(float) if f"_w_{g}" in df.columns else np.ones(len(df))
            w = base_w * wy
            fin = np.isfinite(answer[g].to_numpy(float))
            if fin.sum():
                w[fin] = w[fin] / w[fin].mean()
            df[f"_w_{g}"] = w
        print(f"⏳ 연도 가중 사용: {yw}  (라벨 있는 행 평균 1 정규화)")

    exclude = (["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm",
                "year_month", "_year", "_fday"] + targets
               + [f"{g}_cf" for g in targets] + [f"{g}_avail_frac" for g in targets]
               + [f"_w_{g}" for g in targets])            # 가중 컬럼은 피처가 아니다
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    joblib.dump(feature_cols, save_dir / "feature_cols.pkl")
    print(f"📦 피처 {len(feature_cols)}개 / 행 {len(df)}개")
    print(f"   연도별 라벨 수: "
          + ", ".join(f"{t}={df.groupby('_year')[t].count().to_dict()}" for t in targets[:1]))

    # ---------------- 누설 없는 CV ----------------
    force_prefix = tuple(p.strip() for p in args.force_prefix.split(",") if p.strip())
    if args.featcols_from:
        print(f"📌 피처 선택 고정: {args.featcols_from}/featcols_*.pkl 재사용 "
              f"(top-K 재선택 안 함)")
    if force_prefix:
        n_forced = len([c for c in feature_cols if c.startswith(force_prefix)])
        print(f"📌 강제 포함 접두사 {force_prefix} → 컬럼 {n_forced}개")
    oof, fold_df, _, fold_feats = run_cv(df, feature_cols, targets,
                             config.get("model_type", "XGBoost"), model_params,
                             scheme=args.scheme, n_splits=config.get("n_splits", 5),
                             es_rounds=config.get("early_stopping_rounds", 50),
                             es_mode=args.es_mode, seed=config["seed"],
                             save_dir=str(save_dir), top_k=args.top_k,
                             weight_col=weight_col,
                             featcols_from=args.featcols_from,
                             force_prefix=force_prefix)
    if args.top_k or args.featcols_from:
        nf = sorted({len(v) for v in fold_feats.values()})
        print(f"🌿 폴드별 피처 실제 {nf}개. featcols_*.pkl 저장됨")
    joint_groups = [g.strip() for g in args.joint_groups.split(",") if g.strip()]
    if joint_groups:
        bad = [g for g in joint_groups if g not in targets]
        if bad:
            raise ValueError(f"--joint-groups 에 없는 그룹: {bad}")
        fold_fn = None
        if args.wgrid:
            from src.grids_weighted import make_fold_feature_fn, OUT_COLS
            fold_fn = make_fold_feature_fn(targets)
            print(f"🧭 A-4 격자 가중: 통합 경로에 폴드별·그룹별 {OUT_COLS} 추가 "
                  f"(독립 경로 G2 는 손대지 않음 = 무손상 대조군)")
        oof_j, _, _ = run_cv_joint(df, feature_cols, targets,
                                   config.get("model_type", "XGBoost"), model_params,
                                   scheme=args.scheme, es_mode=args.es_mode,
                                   es_rounds=config.get("early_stopping_rounds", 50),
                                   seed=config["seed"], save_dir=str(save_dir),
                                   top_k=args.top_k, weight_col=weight_col,
                                   featcols_from=args.featcols_from,
                                   force_prefix=force_prefix,
                                   fold_feature_fn=fold_fn)
        for g in joint_groups:
            oof[g] = oof_j[g]
        json.dump(joint_groups, open(save_dir / "joint_groups.json", "w"))
        print(f"🔗 통합 모델 예측 사용: {', '.join(joint_groups)}  "
              f"(나머지는 독립 모델)")

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

    # ---------------- 게이트 (§3.6 규칙 2·6) ----------------
    gate_ok = None
    if args.baseline_oof:
        prev = (pd.read_csv(args.baseline_oof, index_col=0)
                  .reindex(index=oof.index, columns=targets).astype(float))
        print(f"\n🚧 [게이트] raw OOF 짝비교 vs {args.baseline_oof}")
        r = is_difference_real(df, answer, oof, prev, targets,
                               name_a=config.get("version", "new"), name_b="baseline")
        gpos = sum(int(group_score(answer[g].to_numpy(float), oof[g].to_numpy(float),
                                   CAPACITY_KWH[g])[0]
                       > group_score(answer[g].to_numpy(float), prev[g].to_numpy(float),
                                     CAPACITY_KWH[g])[0]) for g in targets)
        gate_ok = bool(r and r["real"] and r["mean"] > 0 and gpos >= 2)
        print(f"   그룹 양수 {gpos}/{len(targets)}")
        if gate_ok:
            print("   ✅ 게이트 통과 — 제출 후보")
        else:
            print("\n" + "🛑" * 34)
            print("🛑  게이트 미달 — raw OOF 가 기준선을 유의하게 넘지 못함")
            print("🛑  §3.6 규칙 2·6: 부호 3/3 ∧ 양수 ∧ |평균|>표준편차 ∧ 그룹 2개 이상")
            print("🛑  제출하지 말 것. 아래 후처리 점수는 §3.6 이 폐기한 지표임")
            print("🛑" * 34 + "\n")

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
               "top_k": args.top_k, "joint_groups": joint_groups,
               "featcols_from": args.featcols_from,
               "force_prefix": list(force_prefix), "wgrid": bool(args.wgrid),
               "objective": model_params.get("objective"),
               "sample_weight": args.sample_weight, "gate_passed": gate_ok, 
               "baseline_oof": args.baseline_oof,  

               "low_weight": args.low_weight,
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
        objective=model_params.get("objective", "-"),
        sample_weight=args.sample_weight,
        raw_oof=raw[0], year_spread=spread, n_features=len(feature_cols),
        features_summary=f"{len(feature_cols)} feats"
                         + (" / clean_target(cf)" if args.clean_target else ""),
        notes=f"raw {raw[0]:.4f} -> nested post {nst[0]:.4f}")


if __name__ == "__main__":
    main()