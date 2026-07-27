"""
scripts/step7_scada_wind_stack.py
==============================================================================
2단계 모델 — SCADA 나셀 풍속을 중간 타깃으로 (스태킹)

왜 이게 다른가
  지금까지 실패한 물리 피처(배치1)는 전부 기존 컬럼의 단조변환이라 트리에
  정보를 못 줬다(§3.2). SCADA 나셀 풍속은 다르다:

  1) **라벨에 없는 정보다.** 26,304시간 × 17터빈 = 44만 관측.
     발전량 라벨(26,304행)보다 훨씬 많고, 파워커브 비선형이 빠진 깔끔한 회귀 타깃이다.
  2) **처음으로 그룹을 구분하는 기상 정보다.** 지금까지 NWP 피처는 세 그룹에 전부
     동일했고 구분은 one-hot 뿐이었다. 실측 나셀 풍속은 그룹마다 실제로 다르다.
  3) NWP 의 **시간별** 편향을 풍속 단계에서 교정하므로 예측의 순서가 바뀐다.
     후처리(최종 출력의 단조변환)가 구조적으로 못 하는 일이다.
  => §3.5 기준 명백한 '정보형 변화'.

구조 (스태킹)
  1단계: NWP 피처 → 그룹별 시간평균 나셀풍속 / 시간내 표준편차(난류 대리)
         LOYO OOF 로 예측해 누설을 막는다. long-format(그룹 one-hot)으로 1개 모델.
  2단계: 1단계 OOF 예측을 **피처로 추가**해 기존 발전량 모델을 학습.

  순수 2단계(풍속 예측 → 파워커브)가 아니라 스태킹인 이유:
  오차가 누적되지 않고, 모델이 쓸지 말지 스스로 정할 수 있다.

규칙 준수
  SCADA 는 학습기간에만 존재한다. 1단계 모델을 학습기간에서 적합해두면
  추론 시에는 NWP 만 입력으로 넣는다. 예측기준시점 이후 정보는 쓰지 않는다.

실행:
    python scripts/step7_scada_wind_stack.py --config configs/config_v13.yaml
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
from src.scada import load_scada, ALIGN_MIN, GROUP_TURBINES, _cols
from src.utils import seed_everything, CAPACITY_KWH
from src.validation import (quiet_warnings, add_time_keys, make_folds, fit_no_leak,
                            run_cv, run_cv_joint, total_score, group_score,
                            score_by_year, is_difference_real, nested_postprocess_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
JOINT = ["kpx_group_1", "kpx_group_3"]          # v13 에서 채택한 조합


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


def scada_group_wind(train_dir):
    """
    그룹별 시간 단위 나셀 풍속 통계.
      ws_mean : 그룹 내 터빈 평균 풍속 (시간평균)
      ws_std  : 시간 내 10분값의 표준편차 (난류 대리)
      ws_disp : 같은 시각 터빈 간 표준편차 (공간 변동)
    """
    out = None
    for maker, fn in [("vestas", "scada_vestas_train.csv"), ("unison", "scada_unison_train.csv")]:
        sc = load_scada(str(Path(train_dir) / fn), maker)
        sc["_h"] = (sc["kst_dtm"] + pd.Timedelta(minutes=ALIGN_MIN[maker])).dt.floor("h")
        for g, turbs in GROUP_TURBINES.items():
            wcols = [_cols(m, i)[1] for m, i in turbs if m == maker and _cols(m, i)[1] in sc.columns]
            if not wcols:
                continue
            tmean = sc[wcols].mean(axis=1)                     # 시각별 터빈 평균
            tdisp = sc[wcols].std(axis=1)                      # 시각별 터빈 간 산포
            tmp = pd.DataFrame({"_h": sc["_h"], "m": tmean, "d": tdisp})
            agg = tmp.groupby("_h").agg(ws_mean=("m", "mean"), ws_std=("m", "std"),
                                        ws_disp=("d", "mean"), n=("m", "size"))
            agg = agg[agg["n"] == 6].drop(columns="n")
            agg.columns = [f"{c}_{g[-1]}" for c in agg.columns]
            out = agg if out is None else out.join(agg, how="outer")
    return out.reset_index().rename(columns={"_h": "forecast_kst_dtm"})


def stage1_wind(df, feats, targets, cfg, verbose=True):
    """1단계: NWP → 나셀 풍속. long-format 1개 모델로 3그룹 동시 학습, LOYO OOF."""
    mtype, mparams = cfg.get("model_type", "XGBoost"), cfg.get("model_params", {})
    # 풍속은 0 근처 점질량이 없으므로 tweedie 대신 제곱오차가 적절.
    # objective 를 바꿀 때 tweedie 전용 파라미터를 같이 빼야 한다.
    # 안 그러면 XGBoost 가 "Parameters: { tweedie_variance_power } are not used" 경고를 뱉는다.
    mp = dict(mparams)
    mp["objective"] = "reg:squarederror"
    for k in ("tweedie_variance_power", "eval_metric", "aft_loss_distribution",
              "aft_loss_distribution_scale", "huber_slope", "quantile_alpha"):
        mp.pop(k, None)
    mp["eval_metric"] = "rmse"
    esr, seed = cfg.get("early_stopping_rounds", 50), cfg["seed"]
    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()

    new = {}
    for stat in ["ws_mean", "ws_std", "ws_disp"]:
        parts = []
        for gi, g in enumerate(targets):
            col = f"{stat}_{g[-1]}"
            if col not in df.columns:
                continue
            d = df[feats].astype(np.float32).copy()
            for k in range(3):
                d[f"g{k}"] = np.float32(1.0 if gi == k else 0.0)
            d["_y"] = df[col].to_numpy(float)
            d["_year"] = yrs; d["_fday"] = fds
            d["_g"] = g; d["_row"] = np.arange(len(df))
            parts.append(d)
        if not parts:
            continue
        L = pd.concat(parts, ignore_index=True)
        lf = [c for c in L.columns if not c.startswith("_")]
        fm = np.isfinite(L["_y"].to_numpy())
        pred = np.full(len(L), np.nan)
        for tr, va, name in make_folds(L["_year"].to_numpy(), L["_fday"].to_numpy(), scheme="loyo"):
            trf = tr[fm[tr]]
            if len(trf) < 500:
                continue
            m, _ = fit_no_leak(mtype, mp, L[lf].iloc[trf], L["_y"].iloc[trf],
                               L["_fday"].to_numpy()[trf], es_rounds=esr, mode="refit", seed=seed)
            pred[va] = m.predict(L[lf].iloc[va])
        L["_p"] = pred
        for g in targets:
            sub = L[L["_g"] == g]
            v = np.full(len(df), np.nan)
            v[sub["_row"].to_numpy()] = sub["_p"].to_numpy()
            new[f"pred_{stat}_{g[-1]}"] = v
        if verbose:
            ok = np.isfinite(L["_y"]) & np.isfinite(L["_p"])
            r = np.corrcoef(L.loc[ok, "_y"], L.loc[ok, "_p"])[0, 1]
            mae = np.abs(L.loc[ok, "_y"] - L.loc[ok, "_p"]).mean()
            print(f"   {stat:8s}: OOF 상관 {r:.4f}  MAE {mae:.3f}  (n={int(ok.sum()):,})")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step7")
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1/5  데이터 + 피처 + 정제 타깃")
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
    feats0 = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    print(f"  행 {len(df):,}  피처 {len(feats0)}개")

    print(BAR); print("STEP 2/5  SCADA 나셀 풍속 집계")
    sw = scada_group_wind(tdir)
    df = df.merge(sw, on="forecast_kst_dtm", how="left")
    for g in targets:
        c = f"ws_mean_{g[-1]}"
        if c in df:
            print(f"  {c}: 유효 {int(df[c].notna().sum()):,}행  "
                  f"평균 {df[c].mean():.2f} m/s  최대 {df[c].max():.1f}")
    if all(f"ws_mean_{g[-1]}" in df for g in targets):
        d13 = (df["ws_mean_1"] - df["ws_mean_3"]).abs()
        print(f"  G1−G3 풍속차: 평균 {d13.mean():.2f} m/s (같은 시각 두 그룹의 실측 차이)")
        print("   -> NWP 는 세 그룹에 동일한 값을 준다. 이게 처음으로 그룹을 가르는 기상 정보다.")

    print(BAR); print("STEP 3/5  1단계 — NWP → 나셀 풍속 (LOYO OOF)")
    new = stage1_wind(df, feats0[:args.top_k] if args.top_k else feats0, targets, cfg)
    for k, v in new.items():
        df[k] = v
    # 파생: NWP 대비 편향 (비율/차이는 단조변환이 아니라 상호작용이다)
    for g in targets:
        pw = f"pred_ws_mean_{g[-1]}"
        if pw in df and "ldaps_v117" in df:
            df[f"bias_{g[-1]}_vs_ldaps"] = df[pw] - df["ldaps_v117"]
        if pw in df and "gfs_v117_powerlaw" in df:
            df[f"bias_{g[-1]}_vs_gfs"] = df[pw] - df["gfs_v117_powerlaw"]
    if all(f"pred_ws_mean_{g[-1]}" in df for g in targets):
        df["pred_ws_g1_minus_g3"] = df["pred_ws_mean_1"] - df["pred_ws_mean_3"]
        df["pred_ws_g1_minus_g2"] = df["pred_ws_mean_1"] - df["pred_ws_mean_2"]
    stage1_cols = [c for c in df.columns if c.startswith(("pred_ws", "bias_"))]
    print(f"  추가 피처 {len(stage1_cols)}개: {stage1_cols[:6]} ...")

    print(BAR); print("STEP 4/5  2단계 A/B — v13 구성(G1·G3 통합 + G2 독립) 고정")
    common = dict(model_type=cfg.get("model_type", "XGBoost"),
                  model_params=cfg.get("model_params", {}), scheme="loyo",
                  es_mode="refit", es_rounds=cfg.get("early_stopping_rounds", 50),
                  seed=cfg["seed"], verbose=False)
    oofs = {}
    for cond, fl in [("A v13", feats0), ("B +풍속스택", feats0 + stage1_cols)]:
        o, _, _, _ = run_cv(df, fl, targets, top_k=args.top_k, **common)
        oj, _, _ = run_cv_joint(df, fl, targets, common["model_type"], common["model_params"],
                                scheme="loyo", es_mode="refit",
                                es_rounds=common["es_rounds"], seed=common["seed"],
                                top_k=args.top_k, verbose=False)
        for g in JOINT:
            o[g] = oj[g]
        for g in targets:
            o[g] = np.clip(o[g] * scale[g], 0, CAPACITY_KWH[g])
        oofs[cond] = o
        s = total_score(answer, o, targets)
        print(f"  {cond:12s}: Total {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}  ({time.time()-t0:.0f}s)")

    print(BAR); print("STEP 5/5  판정 (raw OOF 기준 — §3.6)")
    print(f"  {'조건':14s}" + "".join(f"{g.replace('kpx_','') :>13s}" for g in targets) + f"{'평균':>11s}")
    for n, o in oofs.items():
        ss = [group_score(answer[g].to_numpy(float), o[g].to_numpy(float), CAPACITY_KWH[g])[0]
              for g in targets]
        print(f"  {n:14s}" + "".join(f"{v:13.4f}" for v in ss) + f"{np.mean(ss):11.4f}")
    print()
    for n, o in oofs.items():
        score_by_year(df, answer, o, targets, label=n)
    print()
    is_difference_real(df, answer, oofs["B +풍속스택"], oofs["A v13"], targets,
                       name_a="B(+풍속스택)", name_b="A(v13)")
    print("\n  [그룹별 짝비교]")
    yrs = df["_year"].to_numpy()
    for g in targets:
        vals = []
        for y in sorted(pd.unique(yrs)):
            m = (yrs == y) & answer[g].notna().to_numpy()
            if m.sum() < 200:
                continue
            sa = group_score(answer.loc[m, g].to_numpy(float), oofs["A v13"].loc[m, g].to_numpy(float), CAPACITY_KWH[g])[0]
            sb = group_score(answer.loc[m, g].to_numpy(float), oofs["B +풍속스택"].loc[m, g].to_numpy(float), CAPACITY_KWH[g])[0]
            vals.append(sb - sa)
        sign = "3/3" if all(v > 0 for v in vals) else ("0/3" if all(v < 0 for v in vals) else "혼재")
        print(f"    {g}: " + "  ".join(f"{v:+.4f}" for v in vals) + f"   평균 {np.mean(vals):+.4f}  ({sign})")

    print(BAR); print("(참고) 후처리 후 — §3.6 에 따라 보조 지표로만")
    for n, o in oofs.items():
        print(f"  [{n}]")
        nested_postprocess_score(df, answer, o, targets, optimize_postprocessing,
                                 apply_postprocessing, mode="piecewise", verbose=True)
    for n, o in oofs.items():
        o.to_csv(odir / f"oof_{n.split()[0]}.csv")
    df[["forecast_kst_dtm"] + stage1_cols].to_csv(odir / "stage1_features.csv",
                                                  index=False, encoding="utf-8-sig")
    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()