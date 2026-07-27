"""
scripts/step5_joint_group_ab.py
==============================================================================
3그룹 통합 학습 A/B — G3 격차를 구조로 메운다

배경 (strategy §3.6)
  step4 폴드 점수: G1 ≈ 0.635, G2 ≈ 0.660, **G3 ≈ 0.588**
  최종 점수는 세 그룹 평균이므로 G3 를 G1 수준으로만 올려도 총점 +0.016 이다.
  지금까지 v6→v12 로 번 전체(+0.0079)의 두 배.

  G3 가 약한 이유 중 고칠 수 있는 것은 라벨 부족뿐이다.
  17,538행 (G1/G2 26,200행). EDA 에서 확인한 그룹 간 CCF r=0.90~0.95 (lag 0) 를
  근거로, long-format(행 = 시각 × 그룹) 통합 학습이면 G3 의 실질 학습량이 3배가 된다.

  §3.5 기준으로 이것은 '정보형 변화' 라 OOF→LB 전이도 기대할 수 있다.

조건
  A : 그룹별 독립 모델 3개 (현행 v12 방식)
  B : long-format 통합 모델 1개, 자연 가중
  C : long-format 통합 모델 1개, 그룹 균형 가중 (G3 행에 26200/17538 ≈ 1.49배)

타깃은 세 조건 모두 정제 타깃(cf × 21600, kWh 스케일)으로 통일한다.
평가는 원본 kWh 라벨 기준.

실행:
    python scripts/step5_joint_group_ab.py --config configs/config_v12.yaml
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
                            total_score, score_by_year, is_difference_real,
                            nested_postprocess_score, group_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
CREF = 21600.0                        # 공통 스케일 (그룹 간 용량차 3% 는 무시)

# info.xlsx 기반 그룹 정적 스펙
GROUP_SPEC = {
    "kpx_group_1": dict(gid=0, is_vestas=1, n_turb=6, rotor_d=126.0, cap=21600.0),
    "kpx_group_2": dict(gid=1, is_vestas=1, n_turb=6, rotor_d=126.0, cap=21600.0),
    "kpx_group_3": dict(gid=2, is_vestas=0, n_turb=5, rotor_d=136.0, cap=21000.0),
}


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


def build_long(df, feats, targets):
    """행 = 시각 × 그룹. 그룹 구분은 one-hot + 정적 스펙으로만 준다."""
    parts = []
    for g in targets:
        sp = GROUP_SPEC[g]
        d = df[feats].astype(np.float32).copy()
        for k in range(3):
            d[f"grp_{k}"] = np.float32(1.0 if sp["gid"] == k else 0.0)
        d["grp_is_vestas"] = np.float32(sp["is_vestas"])
        d["grp_n_turb"] = np.float32(sp["n_turb"])
        d["grp_rotor_d"] = np.float32(sp["rotor_d"])
        d["_y"] = df[g].to_numpy(float)
        d["_year"] = df["_year"].to_numpy()
        d["_fday"] = df["_fday"].to_numpy()
        d["_gname"] = g
        d["_row"] = np.arange(len(df))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step5")
    ap.add_argument("--top-k", type=int, default=None,
                    help="메모리가 부족하면 지정 (미지정 시 전체 피처로 가장 깨끗한 비교)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1/4  데이터 + 피처 + 정제 타깃")
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
        tgt = df[f"{g}_cf"].to_numpy(float) * CREF          # 공통 스케일
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * CREF)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * CREF)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets])
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    if args.top_k:
        feats = feats[:args.top_k]
    print(f"  행 {len(df):,}  피처 {len(feats)}개")
    print("  라벨 수: " + ", ".join(f"{g}={int(answer[g].notna().sum()):,}" for g in targets))
    print("  환산 배율: " + ", ".join(f"{g}={scale[g]:.4f}" for g in targets))

    mtype, mparams = cfg.get("model_type", "XGBoost"), cfg.get("model_params", {})
    esr, seed = cfg.get("early_stopping_rounds", 50), cfg["seed"]
    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
    folds = make_folds(yrs, fds, scheme="loyo")

    # ---------------- A: 그룹별 독립 ----------------
    print(BAR); print("STEP 2/4  A — 그룹별 독립 모델 3개 (현행)")
    oof_a = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    for g in targets:
        fm = df[g].notna().to_numpy()
        X, y = df[feats], df[g]
        pred = np.full(len(df), np.nan)
        for tr, va, name in folds:
            trf = tr[fm[tr]]
            m, _ = fit_no_leak(mtype, mparams, X.iloc[trf], y.iloc[trf], fds[trf],
                               es_rounds=esr, mode="refit", seed=seed)
            pred[va] = np.clip(m.predict(X.iloc[va]), 0, 1.15 * CREF)
        oof_a[g] = np.clip(pred * scale[g], 0, CAPACITY_KWH[g])
    sa = total_score(answer, oof_a, targets)
    print(f"  Total {sa[0]:.4f}   1-NMAE {sa[1]:.4f}   FICR {sa[2]:.4f}   ({time.time()-t0:.0f}s)")

    # ---------------- B / C: long-format 통합 ----------------
    print(BAR); print("STEP 3/4  B·C — long-format 통합 모델")
    long = build_long(df, feats, targets)
    lfeats = [c for c in long.columns if not c.startswith("_")]
    print(f"  long 행 {len(long):,}  피처 {len(lfeats)}개 (그룹 one-hot 3 + 정적 3 포함)")
    n_by_g = {g: int(answer[g].notna().sum()) for g in targets}
    bal = {g: max(n_by_g.values()) / n_by_g[g] for g in targets}
    print("  균형 가중치(C): " + ", ".join(f"{g}={bal[g]:.2f}" for g in targets))

    oofs = {"A 독립": oof_a}
    for cond, use_w in [("B 통합", False), ("C 통합+균형", True)]:
        o = pd.DataFrame(index=df.index, columns=targets, dtype=float)
        fm = np.isfinite(long["_y"].to_numpy())
        Xl, yl = long[lfeats], long["_y"]
        lyr, lfd = long["_year"].to_numpy(), long["_fday"].to_numpy()
        wt = long["_gname"].map(bal).to_numpy() if use_w else None
        pred = np.full(len(long), np.nan)
        for tr, va, name in make_folds(lyr, lfd, scheme="loyo"):
            trf = tr[fm[tr]]
            m, _ = fit_no_leak(mtype, mparams, Xl.iloc[trf], yl.iloc[trf], lfd[trf],
                               es_rounds=esr, mode="refit", seed=seed,
                               sample_weight=None if wt is None else wt[trf])
            pred[va] = np.clip(m.predict(Xl.iloc[va]), 0, 1.15 * CREF)
        long["_p"] = pred
        for g in targets:                                  # long -> wide 되돌리기
            sub = long[long["_gname"] == g]
            v = np.full(len(df), np.nan)
            v[sub["_row"].to_numpy()] = sub["_p"].to_numpy()
            o[g] = np.clip(v * scale[g], 0, CAPACITY_KWH[g])
        oofs[cond] = o
        s = total_score(answer, o, targets)
        print(f"  {cond}: Total {s[0]:.4f}   1-NMAE {s[1]:.4f}   FICR {s[2]:.4f}   ({time.time()-t0:.0f}s)")

    print(BAR); print("STEP 4/4  판정")
    print("  [그룹별 점수 — G3 가 실제로 올라갔나]")
    print(f"    {'조건':14s}" + "".join(f"{g.replace('kpx_','') :>14s}" for g in targets) + f"{'평균':>10s}")
    for n, o in oofs.items():
        ss = [group_score(answer[g].to_numpy(float), o[g].to_numpy(float), CAPACITY_KWH[g])[0]
              for g in targets]
        print(f"    {n:14s}" + "".join(f"{v:14.4f}" for v in ss) + f"{np.mean(ss):10.4f}")
    print()
    for n, o in oofs.items():
        score_by_year(df, answer, o, targets, label=n)
    print()
    for n, o in oofs.items():
        if n == "A 독립":
            continue
        print(f"  --- {n} vs A 독립 ---")
        is_difference_real(df, answer, o, oofs["A 독립"], targets, name_a=n, name_b="A 독립")

    best = max(oofs, key=lambda t: total_score(answer, oofs[t], targets)[0])
    print(BAR); print(f"후처리 후 (A vs 최고 {best})")
    for n in dict.fromkeys(["A 독립", best]):
        print(f"  [{n}]")
        nested_postprocess_score(df, answer, oofs[n], targets, optimize_postprocessing,
                                 apply_postprocessing, mode="piecewise", verbose=True)
    for n, o in oofs.items():
        o.to_csv(odir / f"oof_{n.split()[0]}.csv")
    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()