"""
scripts/step6_drift_check.py
==============================================================================
연도 간 표류(drift) 진단 — LOYO 가 이 대회에 적절한 검증인가?

문제 제기
  실제 과제는 2022~2024 로 학습해 2025 를 '외삽' 하는 것이다.
  그런데 LOYO 3폴드 중 그 구조와 같은 것은 val2024 하나뿐이고
  val2022 / val2023 은 내삽이다. 시간에 따라 무언가 표류한다면
  LOYO 는 난이도를 과소평가한다.

  표류 후보(기후 추세보다 훨씬 큰 것들):
    - 수치예보 모델 업그레이드 (GFS/LDAPS 판올림 시 예보→실제 관계가 이동)
    - 터빈 노후화, 제어 소프트웨어 변경, 정비 정책 변화
    - 가용률 변화

측정 방법
  한 해로만 학습해서 다른 해들을 예측한다. 표류가 있다면
  **시간 거리가 멀수록 오차가 커야 한다.**

    학습 2022 -> 예측 2023(거리1), 2024(거리2)
    학습 2023 -> 예측 2022(거리1 역), 2024(거리1 순)
    학습 2024 -> 예측 2023(거리1), 2022(거리2)

  거리 2 가 거리 1 보다 계통적으로 나쁘면 표류 있음.
  같은 거리에서 순방향(미래 예측)이 역방향보다 나쁘면 방향성 표류.

실행:
    python scripts/step6_drift_check.py --config configs/config_v13.yaml
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
from src.validation import quiet_warnings, add_time_keys, fit_no_leak, group_score

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
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1/3  데이터")
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
    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
    years = sorted(pd.unique(yrs))
    print(f"  행 {len(df):,}  피처 {len(feats)}개  연도 {years}")
    print("  연도별 행 수: " + ", ".join(f"{y}={int((yrs==y).sum()):,}" for y in years))

    mtype, mparams = cfg.get("model_type", "XGBoost"), cfg.get("model_params", {})
    esr, seed = cfg.get("early_stopping_rounds", 50), cfg["seed"]

    print(BAR); print("STEP 2/3  한 해로 학습 → 다른 해 예측")
    rows = []
    for ty in years:
        for g in targets:
            fm = df[g].notna().to_numpy()
            trm = (yrs == ty) & fm
            if trm.sum() < 2000:
                continue
            idx = np.where(trm)[0]
            cols = feats
            if args.top_k and args.top_k < len(feats):
                m0, _ = fit_no_leak(mtype, mparams, df[feats].iloc[idx], df[g].iloc[idx],
                                    fds[idx], es_rounds=esr, mode="refit", seed=seed)
                imp = np.asarray(m0.feature_importances_, dtype=float)
                cols = [feats[i] for i in np.argsort(-imp)[:args.top_k]]
            m, _ = fit_no_leak(mtype, mparams, df[cols].iloc[idx], df[g].iloc[idx],
                               fds[idx], es_rounds=esr, mode="refit", seed=seed)
            for py in years:
                if py == ty:
                    continue
                tem = (yrs == py) & fm
                if tem.sum() < 2000:
                    continue
                ii = np.where(tem)[0]
                p = np.clip(m.predict(df[cols].iloc[ii]), 0, 1.15 * CAPACITY_KWH[g]) * scale[g]
                s, one_m, f = group_score(answer[g].to_numpy(float)[ii],
                                          np.clip(p, 0, CAPACITY_KWH[g]), CAPACITY_KWH[g])
                rows.append({"train": ty, "pred": py, "group": g, "score": s,
                             "dist": abs(py - ty), "dir": "순방향" if py > ty else "역방향"})
        print(f"  학습 {ty} 완료 ({time.time()-t0:.0f}s)")

    R = pd.DataFrame(rows)
    print(BAR); print("STEP 3/3  판정")
    piv = R.pivot_table(index="train", columns="pred", values="score", aggfunc="mean")
    print("  [학습연도 × 예측연도 평균 점수]")
    print("   " + piv.round(4).to_string().replace("\n", "\n   "))

    print("\n  [시간 거리별]")
    d = R.groupby("dist")["score"].agg(["mean", "std", "count"])
    print("   " + d.round(4).to_string().replace("\n", "\n   "))
    if 1 in d.index and 2 in d.index:
        gap = d.loc[1, "mean"] - d.loc[2, "mean"]
        print(f"\n   거리1 − 거리2 = {gap:+.4f}")
        print(f"   {'⚠️ 거리가 멀수록 나빠진다 → 표류 있음' if gap > 0.005 else '✅ 거리 효과 미미 → 표류 근거 없음'}")

    print("\n  [방향별 (거리 1 만)]")
    d1 = R[R["dist"] == 1].groupby("dir")["score"].agg(["mean", "std", "count"])
    print("   " + d1.round(4).to_string().replace("\n", "\n   "))
    if len(d1) == 2:
        gap = d1.loc["역방향", "mean"] - d1.loc["순방향", "mean"]
        print(f"\n   역방향 − 순방향 = {gap:+.4f}")
        print(f"   {'⚠️ 미래 예측이 더 어렵다 → 방향성 표류' if gap > 0.005 else '✅ 방향 효과 미미'}")

    print("\n  [예측 연도별 — 연도 자체의 난이도]")
    py = R.groupby("pred")["score"].agg(["mean", "std"])
    print("   " + py.round(4).to_string().replace("\n", "\n   "))
    print("\n   -> 특정 해가 학습원과 무관하게 낮으면 그건 표류가 아니라 '그 해가 어려운 것' 이다.")
    print(BAR); print(f"⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()