"""
scripts/step10_avg_and_post_diag.py
==============================================================================
v14 역전 원인 진단 — OOF 와 '제출 절차' 사이의 두 갈래를 분리한다

관측
  step9/v14 raw OOF : A 0.5950 -> D 0.6205   (+0.0255, 연도 3/3 일관, 그룹×연도 7/8 양수)
  실제 LB           : v13 0.641458 -> v14 0.629475  (-0.011983)
  LB 분해           : 1-nMAE -0.0051 / FICR -0.0189   (FICR 이 3.7배 더 크게 깨졌다)

  학습 재현은 정상이다 (step9 D 0.6205 ≈ v14 Raw OOF 0.6193).
  즉 깨진 곳은 '모델' 이 아니라 'OOF 를 재던 절차' 와 '제출을 만드는 절차' 의 차이다.

  그 차이는 정확히 두 개뿐이다.
    [가설 1] 폴드 평균.  OOF 는 행마다 '그 해를 안 본 모델 1개' 의 예측이다.
             그런데 inference.py 는 폴드 모델 3개의 '평균' 으로 2025를 예측한다.
             MAE 모델의 예측은 조건부 중앙값이고, v14 의 이득은 예측을 최빈값 근처로
             모아 FICR 6% 밴드에 밀어넣는 데서 나왔다(이득의 96%가 FICR).
             평균을 내면 그 집중이 정확히 풀린다. Tweedie(평균 예측)에는 무해했던
             절차가 MAE 에는 유해할 수 있다. FICR 이 NMAE 보다 크게 깨진 것과 부합.
    [가설 2] 후처리 전이.  후처리 파라미터는 OOF 3년으로 적합해 2025에 적용한다.
             v14 는 중첩 이득이 +0.0019 밖에 안 됐다(A 는 +0.0219). 이득이 이만큼
             작으면 적합 분산이 이득을 넘길 수 있다. 실제로 v14 는 LB NMAE 도 떨어졌는데
             후처리는 NMAE 를 내주고 FICR 을 사는 절차라 방향이 맞는다.

  두 가설 모두 'OOF 측정에는 없고 제출 경로에만 있는' 요소다. step9 설계의 구멍이었다.

PART 1 — 폴드 평균이 조건별로 손해인가 (중첩 LOYO)
  바깥 연도 Y 를 '가짜 2025' 로 두고, 나머지 2년으로 연도별 모델 2개를 만든 뒤
  Y 를 (a) 각 단일 모델 (b) 두 모델의 평균 으로 예측해 점수를 비교한다.
  Y 는 어떤 모델의 학습에도 안 들어가므로 누설 없음.
  판정: delta = score(평균) - mean(score(단일)).
        A 에서 delta >= 0 인데 D 에서 delta < 0 이면 가설 1 지지.

PART 2 — 후처리가 조건별로 어느 해에 손해인가
  step9 가 저장한 OOF 를 그대로 써서, 연도 하나를 빼고 후처리를 적합해 그 해에 적용한다.
  연도별로 raw 대비 증감을 본다. D 에서 음수인 해가 있으면 가설 2 지지.

실행:
    python scripts/step10_avg_and_post_diag.py --config configs/config_v14.yaml
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
from src.validation import (quiet_warnings, add_time_keys, fit_no_leak,
                            group_score, total_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
CREF = 21600.0

CONDS = [("A T+U", "tweedie", False), ("D L+W", "mae", True)]


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


def make_weight(actual, cap, low_w=0.2):
    a = np.asarray(actual, float)
    w = np.ones(len(a), dtype=float)
    fin = np.isfinite(a)
    scored = fin & (a >= 0.10 * cap)
    if scored.sum() < 100:
        return w
    w[fin] = low_w
    w[scored] = 0.5 + 0.5 * a[scored] / a[scored].mean()
    w[fin] = w[fin] / w[fin].mean()
    return w


def make_params(base, objective):
    p = dict(base)
    if objective == "mae":
        p.pop("tweedie_variance_power", None)
        p["objective"] = "reg:absoluteerror"
        p["eval_metric"] = "mae"
    else:
        p["objective"] = "reg:tweedie"
        p.setdefault("tweedie_variance_power", 1.5)
        p["eval_metric"] = "tweedie-nloglik@1.5"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--step9-dir", default="./saved_models/_ab_step9")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--low-weight", type=float, default=0.2)
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    seed = cfg["seed"]
    esr = cfg.get("early_stopping_rounds", 30)
    mtype = cfg.get("model_type", "XGBoost")
    mparams = cfg.get("model_params", {})
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("데이터 + 피처 + 정제 타깃")
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
    scale, wcols = {}, {}
    for g in targets:
        tgt = df[f"{g}_cf"].to_numpy(float) * CREF
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * CREF)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * CREF)
        wcols[g] = make_weight(answer[g].to_numpy(float), CAPACITY_KWH[g], args.low_weight)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets])
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]
    print(f"  행 {len(df):,}  피처 {len(feats)}개  연도 {years}")

    # ================================================================= PART 1
    print(BAR)
    print("PART 1  폴드 평균이 조건별로 손해인가 (중첩 LOYO)")
    print("  바깥 연도 Y = 가짜 2025. 나머지 2년으로 연도별 모델 2개 -> Y 를 단일/평균으로 예측.")
    print("  주의: 그룹별 독립 모델로만 본다. 평균 절차는 통합 모델 경로에도 똑같이 적용되므로")
    print("        기전 판정에는 영향이 없다.\n")

    rows = []
    for outer in years:
        inner = [y for y in years if y != outer]
        out_m = (yrs == outer)
        for g in targets:
            lab_ok = df[g].notna().to_numpy()
            a_out = answer[g].to_numpy(float)[out_m]
            cap = CAPACITY_KWH[g]
            if np.isfinite(a_out).sum() < 200 or (a_out[np.isfinite(a_out)] >= 0.10 * cap).sum() < 100:
                continue                                  # 그 해 그 그룹 라벨이 없으면 건너뛴다

            # 폴드 학습셋(=inner 전체) 안에서만 피처 선택. 바깥 연도는 절대 안 본다.
            in_m = lab_ok & np.isin(yrs, inner)
            idx_in = np.where(in_m)[0]
            m0, _ = fit_no_leak(mtype, make_params(mparams, "tweedie"),
                                df[feats].iloc[idx_in], df[g].iloc[idx_in], fds[idx_in],
                                es_rounds=esr, mode="refit", seed=seed)
            imp = np.asarray(m0.feature_importances_, dtype=float)
            cols = [feats[i] for i in np.argsort(-imp)[:args.top_k]]

            for cname, obj, use_w in CONDS:
                p = make_params(mparams, obj)
                preds = []
                for iy in inner:
                    tm = lab_ok & (yrs == iy)
                    idx = np.where(tm)[0]
                    if len(idx) < 500:
                        continue
                    sw = wcols[g][idx] if use_w else None
                    m, _ = fit_no_leak(mtype, p, df[cols].iloc[idx], df[g].iloc[idx],
                                       fds[idx], es_rounds=esr, mode="refit",
                                       seed=seed, sample_weight=sw)
                    pv = np.clip(m.predict(df[cols].iloc[np.where(out_m)[0]]), 0, 1.15 * CREF)
                    preds.append(np.clip(pv * scale[g], 0, cap))
                if len(preds) < 2:
                    continue
                s_sing = [group_score(a_out, pv, cap)[0] for pv in preds]
                s_avg = group_score(a_out, np.mean(preds, axis=0), cap)[0]
                f_sing = [group_score(a_out, pv, cap)[2] for pv in preds]
                f_avg = group_score(a_out, np.mean(preds, axis=0), cap)[2]
                rows.append(dict(outer=outer, group=g, cond=cname,
                                 single=float(np.mean(s_sing)), avg=float(s_avg),
                                 delta=float(s_avg - np.mean(s_sing)),
                                 ficr_single=float(np.mean(f_sing)), ficr_avg=float(f_avg),
                                 ficr_delta=float(f_avg - np.mean(f_sing))))
                print(f"  {outer} {g:14s} {cname:6s}  단일 {np.mean(s_sing):.4f} -> "
                      f"평균 {s_avg:.4f}  Δ {s_avg-np.mean(s_sing):+.4f}   "
                      f"(FICR Δ {f_avg-np.mean(f_sing):+.4f})   [{time.time()-t0:.0f}s]")

    r1 = pd.DataFrame(rows)
    print("\n  [요약 — 평균 절차의 순효과]")
    print(f"    {'조건':8s}{'Δ 총점':>12s}{'Δ FICR':>12s}{'양수/전체':>12s}")
    for cname, _, _ in CONDS:
        s = r1[r1.cond == cname]
        if len(s) == 0:
            continue
        print(f"    {cname:8s}{s.delta.mean():+12.4f}{s.ficr_delta.mean():+12.4f}"
              f"{f'{(s.delta>0).sum()}/{len(s)}':>12s}")
    if len(r1):
        dA = r1[r1.cond == "A T+U"].delta.mean()
        dD = r1[r1.cond == "D L+W"].delta.mean()
        print(f"\n    A 대비 D 의 평균 손해: {dD - dA:+.4f}")
        if dD < 0 <= dA:
            print("    => 가설 1 지지. MAE 모델은 폴드 평균에서 손해를 본다.")
        elif dD < dA - 0.003:
            print("    => 가설 1 부분 지지. 방향은 같으나 A 도 손해다.")
        else:
            print("    => 가설 1 기각. 평균 절차는 조건에 관계없이 중립/이득이다.")
        r1.to_csv(Path(args.step9_dir) / "step10_avg.csv", index=False)

    # ================================================================= PART 2
    print(BAR)
    print("PART 2  후처리가 조건별로 어느 해에 손해인가 (step9 OOF 재사용)")
    s9 = Path(args.step9_dir)
    files = {"A T+U": s9 / "oof_A.csv", "D L+W": s9 / "oof_D.csv"}
    if not all(f.exists() for f in files.values()):
        print(f"  ⚠ {s9} 에 oof_A.csv / oof_D.csv 가 없다. step9 를 먼저 실행할 것. 건너뜀.")
    else:
        print(f"    {'조건':8s}{'연도':>8s}{'raw':>10s}{'후처리':>10s}{'Δ':>10s}")
        for cname, path in files.items():
            oof = pd.read_csv(path, index_col=0).reindex(index=df.index, columns=targets)
            deltas = []
            for y in years:
                te = (yrs == y)
                trm = ~te & answer.notna().any(axis=1).to_numpy()
                pp = optimize_postprocessing(answer.loc[trm], oof.loc[trm],
                                             mode="piecewise", verbose=False)
                post = apply_postprocessing(oof.loc[te].copy(), pp)
                sr = total_score(answer.loc[te], oof.loc[te], targets)[0]
                sp = total_score(answer.loc[te], post, targets)[0]
                deltas.append(sp - sr)
                flag = "  <- 손해" if sp < sr else ""
                print(f"    {cname:8s}{y:>8d}{sr:10.4f}{sp:10.4f}{sp-sr:+10.4f}{flag}")
            d = np.array(deltas)
            print(f"    {cname:8s}{'평균':>8s}{'':10s}{'':10s}{d.mean():+10.4f}"
                  f"   (표준편차 {d.std(ddof=1):.4f})")

    print(BAR)
    print(f"⏱ {time.time()-t0:.0f}초")
    print("다음 행동: 이 진단과 별개로 --no-post 제출 1회를 먼저 태워서 가설 2를 직접 확인할 것.")


if __name__ == "__main__":
    main()