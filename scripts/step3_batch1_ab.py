"""
scripts/step3_batch1_ab.py
==============================================================================
배치 1(물리 핵심 피처) A/B 통제 실험

  A : 기존 features.py 만
  B : 기존 + features_physics.add_batch1_features (허브고도 풍속 재구성 / REWS /
      습윤공기 밀도 / IEC 밀도보정 / SCADA 경험 파워커브)

둘 다 정제 타깃 + LOYO + piecewise 후처리로 동일 조건. 연도별 짝비교로 판정.

추가로 '풍속 추정 진단' 을 낸다. LDAPS 기반 v117 과 GFS 기반 v117 중
어느 쪽이 실제 발전량을 더 잘 설명하는지 전체 기간에서 확인한다.
(실측 head 한 행에서는 LDAPS 9.86 / GFS 3.23 / SCADA 나셀 8.02 였다)

실행:
    python scripts/step3_batch1_ab.py --config configs/config_v11.yaml
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
from src.features_physics import add_batch1_features
from src.utils import seed_everything, CAPACITY_KWH
from src.validation import (quiet_warnings, add_time_keys, run_cv, total_score, score_by_year,
                            is_difference_real, nested_postprocess_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()

BAR = "=" * 76


def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    id_cols = {"forecast_kst_dtm", "grid_id", "data_available_kst_dtm"}
    drop_cols = {"latitude", "longitude"}
    vcols = [c for c in df.columns if c not in (id_cols | drop_cols)]
    piv = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=vcols)
    piv.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in piv.columns]
    agg = df.groupby("forecast_kst_dtm")[vcols].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    out = piv.reset_index().merge(agg.reset_index(), on="forecast_kst_dtm", how="inner")
    if "data_available_kst_dtm" in df.columns:
        av = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first().reset_index()
        out = out.merge(av, on="forecast_kst_dtm", how="left")
    return out


def wind_diagnostic(df, targets):
    """어느 허브고도 풍속 추정이 실제 발전량을 잘 설명하나 (스피어만 기준)."""
    cands = [c for c in ["ldaps_v117", "gfs_v117_3pt", "gfs_v117_powerlaw", "v117_ens",
                         "ldaps_ws50", "ldaps_ws10", "pc_vestas_ldaps_v117_dcorr",
                         "pc_vestas_v117_ens_dcorr"] if c in df.columns]
    if not cands:
        return
    print("\n  [풍속 추정 진단] 실제 발전량과의 순위상관 (스피어만)")
    print(f"    {'추정':32s} " + "  ".join(f"{t.replace('kpx_','') :>12s}" for t in targets)
          + "     중앙값/평균")
    rows = []
    for c in cands:
        rs = []
        for t in targets:
            m = df[t].notna() & df[c].notna()
            rs.append(df.loc[m, c].corr(df.loc[m, t], method="spearman") if m.sum() > 100 else np.nan)
        rows.append((c, rs))
        print(f"    {c:32s} " + "  ".join(f"{r:12.4f}" for r in rs)
              + f"     {np.nanmean(rs):.4f}  {df[c].median():.2f}")
    best = max(rows, key=lambda r: np.nanmean(r[1]))
    print(f"    -> 최고: {best[0]} (평균 {np.nanmean(best[1]):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step3")
    ap.add_argument("--no-clean-target", action="store_true",
                    help="정제 타깃 없이 원본 라벨로 비교")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1/5  데이터 + 기존 피처(A)")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    base = labels.merge(w, on="forecast_kst_dtm", how="left")
    df_a = add_time_keys(build_full_feature_pipeline(base).replace([np.inf, -np.inf], np.nan))
    print(f"  행 {len(df_a):,}")

    print(BAR); print("STEP 2/5  배치1 피처 추가(B)")
    df_b = add_batch1_features(df_a, scada_dir=args.scada_dir)

    # 정제 타깃 결합
    answer = df_a[targets].copy()
    scale = None
    if not args.no_clean_target:
        ct = pd.read_csv(Path(args.scada_dir) / "clean_target.csv", encoding="utf-8-sig")
        ct["forecast_kst_dtm"] = pd.to_datetime(ct.pop("kst_dtm"))
        for d in (df_a, df_b):
            merged = d.merge(ct, on="forecast_kst_dtm", how="left")
            for c in ct.columns:
                if c != "forecast_kst_dtm":
                    d[c] = merged[c].to_numpy()
        scale = {}
        for g in targets:
            cap = CAPACITY_KWH[g]
            tgt = df_a[f"{g}_cf"].to_numpy(float) * cap
            lab = answer[g].to_numpy(float)
            ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * cap)
            scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
            for d in (df_a, df_b):
                d[g] = np.clip(d[f"{g}_cf"].to_numpy(float) * cap, 0, 1.15 * cap)
        print("  정제 타깃 사용. 환산 배율: " + ", ".join(f"{g}={scale[g]:.4f}" for g in targets))

    wind_diagnostic(df_b.assign(**{t: answer[t] for t in targets}), targets)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets
               + [f"{g}_cf" for g in targets] + [f"{g}_avail_frac" for g in targets])
    fa = [c for c in df_a.select_dtypes(include=[np.number]).columns if c not in excl]
    fb = [c for c in df_b.select_dtypes(include=[np.number]).columns if c not in excl]
    print(f"\n  피처 A {len(fa)}개  →  B {len(fb)}개  (+{len(fb)-len(fa)})")

    common = dict(model_type=cfg.get("model_type", "XGBoost"),
                  model_params=cfg.get("model_params", {}),
                  scheme="loyo", es_mode="refit",
                  es_rounds=cfg.get("early_stopping_rounds", 50),
                  seed=cfg["seed"], verbose=False)

    oofs = {}
    for name, d, f in [("A 기존피처", df_a, fa), ("B +배치1", df_b, fb)]:
        print(BAR); print(f"STEP 3/5  {name} 학습")
        o, _, _ = run_cv(d, f, targets, **common)
        if scale is not None:
            for g in targets:
                o[g] = np.clip(o[g] * scale[g], 0, CAPACITY_KWH[g])
        oofs[name] = o
        s = total_score(answer, o, targets)
        print(f"  Total {s[0]:.4f}   1-NMAE {s[1]:.4f}   FICR {s[2]:.4f}")

    print(BAR); print("STEP 4/5  연도별 점수와 유의성 판정  ← 결론")
    for n, o in oofs.items():
        score_by_year(df_a, answer, o, targets, label=n)
    print()
    is_difference_real(df_a, answer, oofs["B +배치1"], oofs["A 기존피처"], targets,
                       name_a="B(+배치1)", name_b="A(기존)")

    print(BAR); print("STEP 5/5  후처리 후")
    post = {}
    for n, o in oofs.items():
        print(f"  [{n}]")
        post[n], _, _ = nested_postprocess_score(
            df_a, answer, o, targets, optimize_postprocessing, apply_postprocessing,
            mode="piecewise", verbose=True)
    print("\n  [후처리 후 연도별]")
    for n, o in post.items():
        score_by_year(df_a, answer, o, targets, label=n + "+후처리")
    print()
    is_difference_real(df_a, answer, post["B +배치1"], post["A 기존피처"], targets,
                       name_a="B+후처리", name_b="A+후처리")

    for n, o in oofs.items():
        o.to_csv(odir / f"oof_{n.split()[0]}.csv")
    print(BAR); print(f"💾 {odir}/ 저장   ⏱ {time.time()-t0:.0f}초")
    print("화면 그대로 보내주면 다음 단계 정한다.")


if __name__ == "__main__":
    main()