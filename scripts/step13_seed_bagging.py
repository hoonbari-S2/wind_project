"""
scripts/step13_seed_bagging.py
==============================================================================
B-4 시드 배깅 — 예측 분산을 깎아 프리미엄을 사는가, 아니면 후처리와 겹치는가

왜 이 실험인가 (§3.11 + §3.14)
  step10 PART 1 에서 폴드 평균의 이득이 총점(+0.0049)보다 FICR(+0.0068)에서 더 컸다.
  FICR 이 오차 6%/8% 계단함수이므로 **예측 분산을 깎으면 NMAE 보다 FICR 이 먼저 오른다.**
  즉 분산 축소는 정확도 레버가 아니라 **프리미엄 레버**다.

  그런데 step12 가 확인했듯 후처리도 분산 축소(수축) 장치다.
  둘이 같은 일을 하면 v14 처럼 상쇄된다(§3.10). 그래서 이 실험의 핵심 질문은 둘이다.

    Q1. N 을 늘리면 raw FICR 이 raw NMAE 보다 빨리 오르는가? (프리미엄 레버가 맞는가)
    Q2. N 을 늘리면 후처리 여지가 줄어드는가?                (후처리와 겹치는가)

  Q1 이 예 이고 Q2 가 아니오 여야 채택이다. 둘 다 예면 v14 의 재판이다.

설계
  v13 구성 고정: 정제 타깃 / 폴드별 top-200 / G1·G3 통합 + G2 독립 / Tweedie.
  시드만 바꿔 여러 모델을 만들고 예측을 단순 평균한다. 가중은 하지 않는다
  (§3.11: 가중 최적화는 +0.0003±0.0007).

  피처 선택은 시드 42 로 폴드마다 한 번만 하고 전 시드가 공유한다.
  시드마다 다시 고르면 다양성이 늘지만 '분산 축소' 와 '피처 다양성' 이 섞여
  원인 귀속이 안 된다. 순수 분산 축소만 본다. (--per-seed-feats 로 반대도 가능)

  N = 1, 2, 3, 5, 8 은 전부 같은 8회 학습의 부분집합이라 학습은 8번만 한다.

판정 (§3.6)
  raw OOF 연도짝비교로 한다. 중첩 후처리 점수는 판정에 쓰지 않고
  '여지가 줄었는가' 를 보는 용도로만 출력한다.

실행
    python scripts/step13_seed_bagging.py --config configs/config_v13.yaml
    python scripts/step13_seed_bagging.py --config configs/config_v13.yaml --seeds 5
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
                            group_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

try:
    from scipy.stats import norm
    _cdf = norm.cdf
except ImportError:
    from math import erf, sqrt
    _cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))

quiet_warnings()
BAR = "=" * 78
CREF = 21600.0
GROUP_SPEC = {
    "kpx_group_1": dict(gid=0, is_vestas=1, n_turb=6, rotor_d=126.0, cap=21600.0),
    "kpx_group_2": dict(gid=1, is_vestas=1, n_turb=6, rotor_d=126.0, cap=21600.0),
    "kpx_group_3": dict(gid=2, is_vestas=0, n_turb=5, rotor_d=136.0, cap=21000.0),
}
JOINT = ["kpx_group_1", "kpx_group_3"]
SOLO = ["kpx_group_2"]
NGRID = [1, 2, 3, 5, 8]


def implied_ficr(nmae):
    """오차 ~ half-normal(평균=nmae) 가정에서 그 NMAE 가 자연히 함의하는 FICR."""
    s = float(nmae) * np.sqrt(np.pi / 2)
    p6 = 2 * _cdf(0.06 / s) - 1
    p8 = 2 * _cdf(0.08 / s) - 1
    return float(p6 + 0.75 * (p8 - p6))


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
    ap.add_argument("--out-dir", default="./saved_models/_ab_step13")
    ap.add_argument("--seeds", type=int, default=8, help="학습할 시드 개수")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--per-seed-feats", action="store_true",
                    help="시드마다 피처를 다시 고른다 (다양성↑, 원인귀속↓)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    base_seed = cfg["seed"]
    esr = cfg.get("early_stopping_rounds", 30)
    mtype = cfg.get("model_type", "XGBoost")
    mparams = cfg.get("model_params", {})
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])
    ngrid = [n for n in NGRID if n <= args.seeds]
    if args.seeds not in ngrid:
        ngrid.append(args.seeds)

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
        tgt = df[f"{g}_cf"].to_numpy(float) * CREF
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * CREF)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * CREF)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets])
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
    folds = make_folds(yrs, fds, scheme="loyo")
    long = build_long(df, feats, JOINT)
    lfeats_all = [c for c in long.columns if not c.startswith("_")]
    lyr, lfd = long["_year"].to_numpy(), long["_fday"].to_numpy()
    lfolds = make_folds(lyr, lfd, scheme="loyo")
    lfm = np.isfinite(long["_y"].to_numpy())
    print(f"  행 {len(df):,}  피처 {len(feats)}  long {len(long):,}  시드 {args.seeds}개  N={ngrid}")

    print(BAR); print("STEP 2/4  폴드별 top-200 선택 (시드 42 기준, 전 시드 공통)")
    topf = {}
    if not args.per_seed_feats:
        for g in SOLO:
            fm = df[g].notna().to_numpy()
            for fi, (tr, va, _) in enumerate(folds):
                trf = tr[fm[tr]]
                m0, _ = fit_no_leak(mtype, mparams, df[feats].iloc[trf], df[g].iloc[trf],
                                    fds[trf], es_rounds=esr, mode="refit", seed=base_seed)
                imp = pd.Series(m0.feature_importances_, index=feats)
                topf[(g, fi)] = list(imp.sort_values(ascending=False).head(args.top_k).index)
        for fi, (tr, va, _) in enumerate(lfolds):
            trf = tr[lfm[tr]]
            m0, _ = fit_no_leak(mtype, mparams, long[lfeats_all].iloc[trf],
                                long["_y"].iloc[trf], lfd[trf], es_rounds=esr,
                                mode="refit", seed=base_seed)
            imp = pd.Series(m0.feature_importances_, index=lfeats_all)
            keep = list(imp.sort_values(ascending=False).head(args.top_k).index)
            for c in ["grp_0", "grp_1", "grp_2", "grp_is_vestas", "grp_n_turb", "grp_rotor_d"]:
                if c not in keep:
                    keep.append(c)
            topf[("joint", fi)] = keep
        print(f"  완료 ({time.time()-t0:.0f}s)")
    else:
        print("  --per-seed-feats: 시드마다 다시 고른다")

    print(BAR); print(f"STEP 3/4  시드 {args.seeds}개 학습")
    per_seed = []                                    # 각 원소: DataFrame(kWh 스케일 OOF)
    for si in range(args.seeds):
        sd = base_seed + si
        p = dict(mparams); p["random_state"] = sd
        oof = pd.DataFrame(index=df.index, columns=targets, dtype=float)

        for g in SOLO:
            fm = df[g].notna().to_numpy()
            pred = np.full(len(df), np.nan)
            for fi, (tr, va, _) in enumerate(folds):
                cols = topf.get((g, fi), feats)
                trf = tr[fm[tr]]
                if args.per_seed_feats:
                    m0, _ = fit_no_leak(mtype, p, df[feats].iloc[trf], df[g].iloc[trf],
                                        fds[trf], es_rounds=esr, mode="refit", seed=sd)
                    imp = pd.Series(m0.feature_importances_, index=feats)
                    cols = list(imp.sort_values(ascending=False).head(args.top_k).index)
                m, _ = fit_no_leak(mtype, p, df[cols].iloc[trf], df[g].iloc[trf], fds[trf],
                                   es_rounds=esr, mode="refit", seed=sd)
                pred[va] = np.clip(m.predict(df[cols].iloc[va]), 0, 1.15 * CREF)
            oof[g] = np.clip(pred * scale[g], 0, CAPACITY_KWH[g])

        pred = np.full(len(long), np.nan)
        for fi, (tr, va, _) in enumerate(lfolds):
            cols = topf.get(("joint", fi), lfeats_all)
            trf = tr[lfm[tr]]
            if args.per_seed_feats:
                m0, _ = fit_no_leak(mtype, p, long[lfeats_all].iloc[trf], long["_y"].iloc[trf],
                                    lfd[trf], es_rounds=esr, mode="refit", seed=sd)
                imp = pd.Series(m0.feature_importances_, index=lfeats_all)
                cols = list(imp.sort_values(ascending=False).head(args.top_k).index)
                for c in ["grp_0", "grp_1", "grp_2", "grp_is_vestas", "grp_n_turb", "grp_rotor_d"]:
                    if c not in cols:
                        cols.append(c)
            m, _ = fit_no_leak(mtype, p, long[cols].iloc[trf], long["_y"].iloc[trf], lfd[trf],
                               es_rounds=esr, mode="refit", seed=sd)
            pred[va] = np.clip(m.predict(long[cols].iloc[va]), 0, 1.15 * CREF)
        long["_p"] = pred
        for g in JOINT:
            sub = long[long["_gname"] == g]
            v = np.full(len(df), np.nan)
            v[sub["_row"].to_numpy()] = sub["_p"].to_numpy()
            oof[g] = np.clip(v * scale[g], 0, CAPACITY_KWH[g])

        per_seed.append(oof)
        oof.to_csv(odir / f"oof_seed{sd}.csv")
        s = total_score(answer, oof, targets)
        print(f"  seed {sd}: raw {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}   ({time.time()-t0:.0f}s)")

    # 시드 간 예측 산포 — 배깅이 실제로 깎고 있는 양
    print(BAR); print("시드 간 예측 산포 (배깅이 깎는 대상)")
    for g in targets:
        M = np.vstack([o[g].to_numpy(float) for o in per_seed])
        sd_row = np.nanstd(M, axis=0, ddof=1) / CAPACITY_KWH[g]
        sd_row = sd_row[np.isfinite(sd_row)]
        print(f"  {g}: 중앙값 {np.median(sd_row)*100:.2f}%cap  "
              f"90분위 {np.quantile(sd_row,0.9)*100:.2f}%cap   (밴드 경계 6%)")

    print(BAR); print("STEP 4/4  N 별 집계")
    print(f"  {'N':>3s}{'raw':>9s}{'1-NMAE':>9s}{'FICR':>9s}{'함의':>9s}{'프리미엄':>10s}"
          f"{'후처리후':>10s}{'여지':>9s}")
    oofs, prem = {}, {}
    for n in ngrid:
        M = {g: np.nanmean([o[g].to_numpy(float) for o in per_seed[:n]], axis=0) for g in targets}
        o = pd.DataFrame(M, index=df.index)
        oofs[n] = o
        s = total_score(answer, o, targets)
        imp = implied_ficr(1 - s[1])
        prem[n] = s[2] - imp
        # 후처리 여지: 그 해를 빼고 적합해 그 해에 적용 (판정용 아님, 겹침 확인용)
        py = []
        for y in [int(v) for v in sorted(pd.unique(yrs)) if (yrs == v).sum() >= 200]:
            te = (yrs == y); tr = ~te & answer.notna().any(axis=1).to_numpy()
            pp = optimize_postprocessing(answer.loc[tr], o.loc[tr], mode="piecewise", verbose=False)
            post = apply_postprocessing(o.loc[te].copy(), pp)
            py.append((total_score(answer.loc[te], o.loc[te], targets)[0],
                       total_score(answer.loc[te], post, targets)[0]))
        r_, p_ = np.mean([a for a, b in py]), np.mean([b for a, b in py])
        print(f"  {n:3d}{s[0]:9.4f}{s[1]:9.4f}{s[2]:9.4f}{imp:9.4f}{prem[n]:+10.4f}"
              f"{p_:10.4f}{p_-r_:9.4f}")
        o.to_csv(odir / f"oof_N{n}.csv")

    n_max = ngrid[-1]
    print(BAR); print("판정")
    s1, sN = total_score(answer, oofs[1], targets), total_score(answer, oofs[n_max], targets)
    d_nmae, d_ficr = sN[1] - s1[1], sN[2] - s1[2]
    print(f"  Q1. 분산 축소가 프리미엄 레버인가 — N=1 -> N={n_max}")
    print(f"      Δ(1-NMAE) {d_nmae:+.4f}   ΔFICR {d_ficr:+.4f}   "
          f"Δ프리미엄 {prem[n_max]-prem[1]:+.4f}")
    print(f"      => {'✅ FICR 이 NMAE 보다 빨리 오른다. 프리미엄 레버가 맞다.' if d_ficr > 2*max(d_nmae,1e-9) else '❌ FICR 이 NMAE 대비 빨리 오르지 않는다. 그냥 정확도 개선이다.'}")
    print()
    print("  Q2. 후처리와 겹치는가 — 위 표의 '여지' 열이 N 에 따라 줄면 겹치는 것이다.")
    print("      (v14 는 여지가 0.0236 -> 0.0026 으로 붕괴해 총점이 뒤집혔다. §3.10)")
    print()
    print("  [연도별 raw]")
    for n in ngrid:
        score_by_year(df, answer, oofs[n], targets, label=f"N={n}")
    print()
    print(f"  [연도짝비교 — N={n_max} vs N=1, raw 기준. 이것이 판정이다 (§3.6)]")
    is_difference_real(df, answer, oofs[n_max], oofs[1], targets,
                       name_a=f"N={n_max}", name_b="N=1")
    print()
    print("  [그룹별 raw]")
    print(f"    {'N':>3s}" + "".join(f"{g.replace('kpx_',''):>14s}" for g in targets))
    for n in ngrid:
        ss = [group_score(answer[g].to_numpy(float), oofs[n][g].to_numpy(float),
                          CAPACITY_KWH[g])[0] for g in targets]
        print(f"    {n:3d}" + "".join(f"{v:14.4f}" for v in ss))

    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")
    print("채택 조건: Q1 ✅ 이고 '여지' 가 크게 안 줄고 연도짝비교가 신호일 것. 셋 다여야 제출한다.")


if __name__ == "__main__":
    main()