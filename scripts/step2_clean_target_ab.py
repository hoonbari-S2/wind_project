"""
scripts/step2_clean_target_ab.py
==============================================================================
정제 타깃이 실제로 NMAE 를 깎는가? — 통제된 A/B 실험

  A (현재 방식) : 원본 라벨 kWh 를 그대로 학습
  B (정제 타깃) : cf = label / (k * 가용용량) 을 학습하고 kWh 로 되돌림

두 조건에서 피처·모델·하이퍼파라미터·시드를 전부 동일하게 두고, LOYO 로
연도별 점수를 낸 뒤 '연도별 짝비교' 로 차이가 노이즈인지 신호인지 판정한다.

실행:
    python scripts/step2_clean_target_ab.py --config configs/config_v6.yaml

주의: saved_models 를 건드리지 않는다 (별도 스크래치 폴더 사용).
==============================================================================
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_full_feature_pipeline
from src.utils import seed_everything, CAPACITY_KWH
from src.validation import (add_time_keys, run_cv, total_score, score_by_year,
                            is_difference_real, nested_postprocess_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

BAR = "=" * 76


def process_weather_data(df, prefix):
    """data_available_kst_dtm 을 살려서 내보낸다 (LOYO 블록 단위 분할에 필요)."""
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    id_cols = {"forecast_kst_dtm", "grid_id", "data_available_kst_dtm"}
    drop_cols = {"latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in (id_cols | drop_cols)]

    piv = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=value_cols)
    piv.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in piv.columns]
    piv = piv.reset_index()
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    out = piv.merge(agg.reset_index(), on="forecast_kst_dtm", how="inner")
    if "data_available_kst_dtm" in df.columns:
        av = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first().reset_index()
        out = out.merge(av, on="forecast_kst_dtm", how="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step2")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1/5  데이터 + 피처")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = labels.merge(w, on="forecast_kst_dtm", how="left")
    df = build_full_feature_pipeline(df).replace([np.inf, -np.inf], np.nan)
    df = add_time_keys(df)

    ct = pd.read_csv(Path(args.scada_dir) / "clean_target.csv", encoding="utf-8-sig")
    ct["forecast_kst_dtm"] = pd.to_datetime(ct.pop("kst_dtm"))
    df = df.merge(ct, on="forecast_kst_dtm", how="left")
    print(f"  행 {len(df):,}  |  정제 타깃 결합 완료")

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets
               + [f"{g}_cf" for g in targets] + [f"{g}_avail_frac" for g in targets])
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    print(f"  피처 {len(feats)}개")

    answer = df[targets].copy()
    common = dict(model_type=cfg.get("model_type", "XGBoost"),
                  model_params=cfg.get("model_params", {}),
                  scheme="loyo", es_mode="refit", seed=cfg["seed"], verbose=False)

    # ---------------- A: 원본 라벨 ----------------
    print(BAR); print("STEP 2/5  조건 A — 원본 라벨 kWh 학습 (현재 방식)")
    oof_a, fold_a, _ = run_cv(df, feats, targets, **common)
    sa = total_score(answer, oof_a, targets)
    print(f"  Total {sa[0]:.4f}   1-NMAE {sa[1]:.4f}   FICR {sa[2]:.4f}")

    # ---------------- B: 정제 타깃 ----------------
    print(BAR); print("STEP 3/5  조건 B — 정제 타깃(cf) 학습 후 kWh 환산")
    df_b = df.copy()
    scale = {}
    for g in targets:
        c = f"{g}_cf"
        if c not in df_b:
            print(f"  ⚠️ {c} 없음 — B 조건 불가"); return
        af = df_b[f"{g}_avail_frac"]
        scale[g] = float(np.nanmean(af)) * CAPACITY_KWH[g]     # cf -> kWh 환산 배율
        df_b[g] = df_b[c]                                       # 타깃 교체
    oof_b_cf, fold_b, _ = run_cv(df_b, feats, targets, **common)
    oof_b = oof_b_cf.copy()
    for g in targets:
        oof_b[g] = np.clip(oof_b_cf[g] * scale[g], 0, CAPACITY_KWH[g])
    sb = total_score(answer, oof_b, targets)
    print(f"  Total {sb[0]:.4f}   1-NMAE {sb[1]:.4f}   FICR {sb[2]:.4f}")
    print(f"  (cf->kWh 환산 배율: " + ", ".join(f"{g}={scale[g]:.0f}" for g in targets) + ")")

    # ---------------- 공통 평가 마스크 ----------------
    # B 는 가용률<50% 행을 학습에서 제외했으므로 그 행의 OOF 가 NaN 이다.
    # A 는 예측이 있으니 그대로 비교하면 A/B 가 서로 다른 행에서 채점된다.
    # 통제 실험이 되도록 양쪽 다 예측이 있는 행으로 맞춘다.
    print(BAR); print("STEP 3.5/5  공통 평가 행 정렬 (통제 조건)")
    for g in targets:
        bad = ~np.isfinite(oof_a[g].to_numpy(float)) | ~np.isfinite(oof_b[g].to_numpy(float))
        lab_ok = df[g].notna().to_numpy()
        drop = int((bad & lab_ok).sum())
        oof_a.loc[bad, g] = np.nan
        oof_b.loc[bad, g] = np.nan
        print(f"  {g}: 라벨은 있는데 한쪽 예측이 없어 제외한 행 {drop:,}  "
              f"(공통 채점 행 {int((~bad & lab_ok).sum()):,})")
    sa = total_score(answer, oof_a, targets)
    sb = total_score(answer, oof_b, targets)
    print(f"  정렬 후  A: {sa[0]:.4f} (1-NMAE {sa[1]:.4f} / FICR {sa[2]:.4f})")
    print(f"          B: {sb[0]:.4f} (1-NMAE {sb[1]:.4f} / FICR {sb[2]:.4f})")

    # ---------------- 판정 ----------------
    print(BAR); print("STEP 4/5  연도별 점수와 유의성 판정  ← 여기가 결론")
    score_by_year(df, answer, oof_a, targets, label="A 원본라벨")
    score_by_year(df, answer, oof_b, targets, label="B 정제타깃")
    print()
    verdict = is_difference_real(df, answer, oof_b, oof_a, targets,
                                 name_a="B(정제)", name_b="A(원본)")

    print(BAR); print("STEP 5/5  후처리 적용 후 (중첩 평가)")
    for name, oof in [("A 원본라벨", oof_a), ("B 정제타깃", oof_b)]:
        print(f"  [{name}]")
        nested_postprocess_score(df, answer, oof, targets,
                                 optimize_postprocessing, apply_postprocessing,
                                 mode="piecewise", verbose=True)

    oof_a.to_csv(odir / "oof_A_raw_label.csv"); oof_b.to_csv(odir / "oof_B_clean_target.csv")
    pd.concat([fold_a.assign(cond="A"), fold_b.assign(cond="B")]).to_csv(
        odir / "fold_scores.csv", index=False)
    print(BAR)
    print(f"💾 {odir}/ 저장 완료   ⏱ {time.time()-t0:.0f}초")
    print("화면 그대로 복사해서 보내주면 다음 단계 정한다.")
    if verdict and not verdict["real"]:
        print("\n⚠️ 판정이 '노이즈' 로 나왔다면 정제 타깃은 이 형태로는 효과가 없다는 뜻이다.")
        print("   버리지 말고 알려줄 것 — 환산 배율이나 min_avail_frac 조정 여지가 있다.")


if __name__ == "__main__":
    main()