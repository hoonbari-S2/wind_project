"""
scripts/step4_prune_sweep.py
==============================================================================
피처 가지치기 스위프 — "더하기" 가 아니라 "빼기" 를 검증한다.

배경
  현재 812개 피처 / 26,304행. colsample_bytree=0.8 이면 트리마다 650개를 본다.
  실제로 의미 있는 게 50~100개라면 나머지는 분할 후보를 희석시킨다.
  지금까지의 실험은 전부 더하는 쪽이었고(배치1 +53개 → 0), 빼는 쪽은 한 번도
  검증하지 않았다.

누설 차단
  피처 선택을 전체 데이터에서 하면 검증폴드를 본 것이 되어 가짜 이득이 나온다.
  (feature selection leakage — 흔한 함정)
  여기서는 **각 LOYO 폴드의 학습셋 안에서만** 중요도를 구하고 상위 K개를 골라
  다시 학습한 뒤 검증폴드를 예측한다. 폴드마다 선택된 피처가 다를 수 있다.

실행:
    python scripts/step4_prune_sweep.py --config configs/config_v11.yaml
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
from src.validation import (quiet_warnings, add_time_keys, make_folds, fit_no_leak,
                            group_score, total_score, score_by_year, is_difference_real,
                            nested_postprocess_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78


def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    idc = {"forecast_kst_dtm", "grid_id", "data_available_kst_dtm"}
    vc = [c for c in df.columns if c not in (idc | {"latitude", "longitude"})]
    piv = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=vc)
    piv.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in piv.columns]
    agg = df.groupby("forecast_kst_dtm")[vc].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    out = piv.reset_index().merge(agg.reset_index(), on="forecast_kst_dtm", how="inner")
    if "data_available_kst_dtm" in df.columns:
        av = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first().reset_index()
        out = out.merge(av, on="forecast_kst_dtm", how="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step4")
    ap.add_argument("--k-list", default="400,200,120,60,30")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])
    Ks = [int(k) for k in args.k_list.split(",")]

    print(BAR); print("STEP 1/4  데이터 + 피처")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = add_time_keys(build_full_feature_pipeline(
        labels.merge(w, on="forecast_kst_dtm", how="left")).replace([np.inf, -np.inf], np.nan))

    answer = df[targets].copy()
    ct = pd.read_csv(Path(args.scada_dir) / "clean_target.csv", encoding="utf-8-sig")
    ct["forecast_kst_dtm"] = pd.to_datetime(ct.pop("kst_dtm"))
    df = df.merge(ct, on="forecast_kst_dtm", how="left")
    scale = {}
    for g in targets:
        cap = CAPACITY_KWH[g]
        tgt = df[f"{g}_cf"].to_numpy(float) * cap
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * cap)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * cap)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets])
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    print(f"  행 {len(df):,}  전체 피처 {len(feats)}개  (정제 타깃 사용)")

    mtype = cfg.get("model_type", "XGBoost")
    mparams = cfg.get("model_params", {})
    es_rounds = cfg.get("early_stopping_rounds", 50)

    print(BAR); print("STEP 2/4  폴드별 중요도 산출 (학습셋 안에서만 — 누설 차단)")
    rank = {}       # (target, fold) -> 중요도 내림차순 피처 리스트
    conc = []
    for tg in targets:
        fm = df[tg].notna().to_numpy()
        X, y = df[feats], df[tg]
        yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
        for tr, va, name in make_folds(yrs, fds, scheme="loyo"):
            trf = tr[fm[tr]]
            m, _ = fit_no_leak(mtype, mparams, X.iloc[trf], y.iloc[trf], fds[trf],
                               es_rounds=es_rounds, mode="refit", seed=cfg["seed"])
            imp = np.asarray(m.feature_importances_, dtype=float)
            order = np.argsort(-imp)
            rank[(tg, name)] = [feats[i] for i in order]
            tot = imp.sum() + 1e-12
            conc.append({"target": tg, "fold": name,
                         **{f"top{k}": imp[order[:k]].sum() / tot for k in Ks},
                         "nonzero": int((imp > 0).sum())})
    cdf = pd.DataFrame(conc)
    print("  중요도(gain) 집중도 — 상위 K개가 전체의 몇 %를 차지하나")
    print("   " + cdf.groupby("target")[[f"top{k}" for k in Ks] + ["nonzero"]]
          .mean().round(3).to_string().replace("\n", "\n   "))

    print(BAR); print("STEP 3/4  K별 재학습")
    oofs = {}
    for K in [len(feats)] + Ks:
        tag = "전체" if K == len(feats) else f"top{K}"
        o = pd.DataFrame(index=df.index, columns=targets, dtype=float)
        for tg in targets:
            fm = df[tg].notna().to_numpy()
            y = df[tg]
            yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
            pred = np.full(len(df), np.nan)
            for tr, va, name in make_folds(yrs, fds, scheme="loyo"):
                cols = rank[(tg, name)][:K]
                trf = tr[fm[tr]]
                m, _ = fit_no_leak(mtype, mparams, df[cols].iloc[trf], y.iloc[trf], fds[trf],
                                   es_rounds=es_rounds, mode="refit", seed=cfg["seed"])
                pred[va] = np.clip(m.predict(df[cols].iloc[va]), 0, 1.15 * CAPACITY_KWH[tg])
            o[tg] = np.clip(pred * scale[tg], 0, CAPACITY_KWH[tg])
        oofs[tag] = o
        s = total_score(answer, o, targets)
        print(f"  {tag:>6}: Total {s[0]:.4f}   1-NMAE {s[1]:.4f}   FICR {s[2]:.4f}   "
              f"({time.time()-t0:.0f}s)")

    print(BAR); print("STEP 4/4  전체 대비 짝비교  ← 결론")
    for tag, o in oofs.items():
        score_by_year(df, answer, o, targets, label=tag)
    print()
    for tag, o in oofs.items():
        if tag == "전체":
            continue
        print(f"  --- {tag} vs 전체 ---")
        is_difference_real(df, answer, o, oofs["전체"], targets, name_a=tag, name_b="전체")

    best = max(oofs, key=lambda t: total_score(answer, oofs[t], targets)[0])
    print(BAR); print(f"후처리 후 (전체 vs 최고 {best})")
    for tag in dict.fromkeys(["전체", best]):
        print(f"  [{tag}]")
        nested_postprocess_score(df, answer, oofs[tag], targets, optimize_postprocessing,
                                 apply_postprocessing, mode="piecewise", verbose=True)

    # 어떤 피처가 살아남았나 (모든 폴드 공통 상위)
    kk = min(Ks)
    from collections import Counter
    cnt = Counter()
    for (tg, name), lst in rank.items():
        cnt.update(lst[:kk])
    print(BAR); print(f"모든 폴드에서 상위 {kk}에 든 피처 (등장 {len(rank)}회 만점)")
    for f, c in cnt.most_common(30):
        print(f"  {c:2d}/{len(rank)}  {f}")
    pd.DataFrame(cnt.most_common(), columns=["feature", "count"]).to_csv(
        odir / "feature_survival.csv", index=False, encoding="utf-8-sig")
    for tag, o in oofs.items():
        o.to_csv(odir / f"oof_{tag}.csv")
    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()