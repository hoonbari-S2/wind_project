"""
scripts/step9_objective_ab.py
==============================================================================
목적함수 · 표본가중 2x2 요인 실험 — 학습 문제를 평가 산식에 맞춘다

배경 (strategy §3.5 / §3.9)
  지금까지 성공한 셋(정제 타깃 / 통합 학습 / 가지치기)은 전부 '학습 문제 자체를
  바꾼 것' 이었고, 실패한 둘(물리 피처 / SCADA 스태킹)은 전부 'NWP 로부터 유도
  가능한 양을 다시 계산한 것' 이었다. NWP 로부터 유도 가능한 것은 이미 모델이
  전부 갖고 있다.

  아직 안 건드린 '학습 문제' 가 목적함수다. 현재:
      학습 목적함수 : reg:tweedie(p=1.5) = 로그우도
      대회 목적함수 : 0.5*(1-NMAE) + 0.5*FICR
  둘은 아무 관계가 없다. 특히 두 가지를 무시하고 있다.

    (1) NMAE 는 용량으로 정규화된 순수 L1 이다. 그룹 안에서 capacity 는 상수이므로
        '그룹별 MAE 최소화' = 'NMAE 최소화' 가 정확히 성립한다.
        Tweedie 로그우도는 L1 이 아니다.
    (2) FICR 은 발전량 가중이고(earned = sum(actual * price)), 채점 자체가
        actual >= 0.10*cap 인 행에서만 일어난다. 지금은 채점되지도 않는
        저발전 행(약 30%)을 나머지와 똑같은 무게로 학습해 모델 용량을 나눠 쓴다.

설계 (2x2 요인)
  목적함수 : T = reg:tweedie(p=1.5, 현행)   |   L = reg:absoluteerror
  가중     : U = 균일(현행)                  |   W = 산식정합 가중

  A = T+U (현행 v13)   B = T+W   C = L+U   D = L+W

  요인설계라 '목적함수 효과' 와 '가중 효과' 와 '상호작용' 을 각각 분리해서 읽는다.
  A/B/C/D 를 늘어놓고 최고를 고르는 게 아니다 (그건 4개 중 최고를 뽑는 사후선택).

가중 정의 (사전 고정, 튜닝 금지)
  a_i = 원본 라벨(실발전량), cap = 그룹 용량
      a_i <  0.10*cap  ->  w_i = low_w          (기본 0.2, 0 이 아닌 이유는 아래)
      a_i >= 0.10*cap  ->  w_i = 0.5 + 0.5 * a_i / mean(a_scored)
  이는 산식 0.5*(1-NMAE) + 0.5*FICR 의 구조를 그대로 옮긴 것이다.
  NMAE 항은 채점행에 균일(0.5), FICR 항은 채점행에 발전량 비례(0.5).
  마지막에 평균이 1이 되도록 정규화한다 (학습률과의 상호작용 제거).

  low_w 를 0 으로 두지 않는 이유: 채점 여부는 '실제값' 이 정하는 것이라 저발전 행을
  아무리 틀려도 점수엔 안 들어가지만, 그 행을 완전히 버리면 임계값(10% cap) 근처의
  캘리브레이션이 무너져 채점행 예측까지 흔들린다. 0.2 는 '거의 무시하되 버리진 않음'.

구성은 v13 채택안(G)을 그대로 고정한다.
  G1, G3 : long-format 통합 모델 1개
  G2     : 독립 모델 1개
  타깃   : 정제 타깃(cf * 21600, kWh 스케일), 평가는 원본 kWh 라벨 기준
  판정   : raw OOF 연도짝비교 (§3.6 — 중첩 후처리 판정은 검정력이 낮다)
           + 그룹별 분해 (§3.7 — 총점 짝비교보다 강력하다)

실행:
    python scripts/step9_objective_ab.py --config configs/config_v13.yaml
    python scripts/step9_objective_ab.py --config configs/config_v13.yaml --top-k 200
==============================================================================
"""
import argparse
import json
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
CREF = 21600.0

GROUP_SPEC = {
    "kpx_group_1": dict(gid=0, is_vestas=1, n_turb=6, rotor_d=126.0, cap=21600.0),
    "kpx_group_2": dict(gid=1, is_vestas=1, n_turb=6, rotor_d=126.0, cap=21600.0),
    "kpx_group_3": dict(gid=2, is_vestas=0, n_turb=5, rotor_d=136.0, cap=21000.0),
}

# v13 채택 구성: G1/G3 통합, G2 독립
JOINT_GROUPS = ["kpx_group_1", "kpx_group_3"]
SOLO_GROUPS = ["kpx_group_2"]

CONDS = [
    ("A T+U(현행)", "tweedie", False),
    ("B T+W", "tweedie", True),
    ("C L+U", "mae", False),
    ("D L+W", "mae", True),
]


# ------------------------------------------------------------------ 데이터
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
    """행 = 시각 x 그룹. 그룹 구분은 one-hot + 정적 스펙으로만 준다."""
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
        d["_w"] = df[f"_w_{g}"].to_numpy(float)
        d["_year"] = df["_year"].to_numpy()
        d["_fday"] = df["_fday"].to_numpy()
        d["_gname"] = g
        d["_row"] = np.arange(len(df))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ 가중
def make_weight(actual, cap, low_w=0.2):
    """
    산식정합 가중.
      채점 제외행(actual < 0.10*cap) -> low_w
      채점 대상행                     -> 0.5 + 0.5 * a / mean(a_scored)
    평균 1로 정규화. 라벨이 없는 행은 어차피 학습에서 빠지므로 1.0 으로 채운다.
    """
    a = np.asarray(actual, float)
    w = np.ones(len(a), dtype=float)
    fin = np.isfinite(a)
    if fin.sum() == 0:
        return w
    scored = fin & (a >= 0.10 * cap)
    if scored.sum() < 100:
        return w
    abar = a[scored].mean()
    w[fin] = low_w
    w[scored] = 0.5 + 0.5 * a[scored] / abar
    w[fin] = w[fin] / w[fin].mean()
    return w


# ------------------------------------------------------------------ 모델
def make_params(base, objective, force_cpu=False):
    p = dict(base)
    if objective == "mae":
        p.pop("tweedie_variance_power", None)
        p["objective"] = "reg:absoluteerror"
        p["eval_metric"] = "mae"
    if force_cpu:
        p["device"] = "cpu"
        p.pop("tree_method", None)
        p["tree_method"] = "hist"
    return p


def fit_safe(mtype, params, X, y, fdays, esr, seed, sw, objective):
    """reg:absoluteerror 는 일부 xgboost 빌드에서 GPU 미지원 -> CPU 폴백."""
    try:
        return fit_no_leak(mtype, params, X, y, fdays, es_rounds=esr,
                           mode="refit", seed=seed, sample_weight=sw)
    except Exception as e:
        if objective != "mae":
            raise
        print(f"      [!] GPU 실패 -> CPU 폴백 ({type(e).__name__}: {str(e)[:70]})")
        p2 = dict(params); p2["device"] = "cpu"
        return fit_no_leak(mtype, p2, X, y, fdays, es_rounds=esr,
                           mode="refit", seed=seed, sample_weight=sw)


def run_condition(name, objective, use_w, df, feats, targets, folds,
                  mtype, mparams, esr, seed, scale, top_feats):
    """v13 구성(G1/G3 통합 + G2 독립)으로 한 조건의 OOF 를 만든다."""
    p = make_params(mparams, objective)
    oof = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    fds = df["_fday"].to_numpy()

    # --- 독립 그룹 ---
    for g in SOLO_GROUPS:
        fm = df[g].notna().to_numpy()
        y = df[g]
        w_all = df[f"_w_{g}"].to_numpy(float) if use_w else None
        pred = np.full(len(df), np.nan)
        for fi, (tr, va, _) in enumerate(folds):
            fcols = top_feats.get((g, fi), feats)
            X = df[fcols]
            trf = tr[fm[tr]]
            sw = None if w_all is None else w_all[trf]
            m, _ = fit_safe(mtype, p, X.iloc[trf], y.iloc[trf], fds[trf],
                            esr, seed, sw, objective)
            pred[va] = np.clip(m.predict(X.iloc[va]), 0, 1.15 * CREF)
        oof[g] = np.clip(pred * scale[g], 0, CAPACITY_KWH[g])

    # --- 통합 그룹 ---
    long = build_long(df, feats, JOINT_GROUPS)
    lfeats_all = [c for c in long.columns if not c.startswith("_")]
    fm = np.isfinite(long["_y"].to_numpy())
    yl = long["_y"]
    lyr, lfd = long["_year"].to_numpy(), long["_fday"].to_numpy()
    wl = long["_w"].to_numpy(float) if use_w else None
    pred = np.full(len(long), np.nan)
    for fi, (tr, va, _) in enumerate(make_folds(lyr, lfd, scheme="loyo")):
        fcols = top_feats.get(("joint", fi), lfeats_all)
        Xl = long[fcols]
        trf = tr[fm[tr]]
        sw = None if wl is None else wl[trf]
        m, _ = fit_safe(mtype, p, Xl.iloc[trf], yl.iloc[trf], lfd[trf],
                        esr, seed, sw, objective)
        pred[va] = np.clip(m.predict(Xl.iloc[va]), 0, 1.15 * CREF)
    long["_p"] = pred
    for g in JOINT_GROUPS:
        sub = long[long["_gname"] == g]
        v = np.full(len(df), np.nan)
        v[sub["_row"].to_numpy()] = sub["_p"].to_numpy()
        oof[g] = np.clip(v * scale[g], 0, CAPACITY_KWH[g])
    return oof


def select_top_feats(df, feats, targets, folds, mtype, mparams, esr, seed, k):
    """
    폴드별 top-k 피처를 '현행 조건(A)' 로 한 번만 뽑아 모든 조건에 공통 적용한다.
    - 중요도는 폴드 학습셋 안에서만 계산 -> 선택 누설 없음
    - 모든 조건이 같은 피처를 보므로 비교에서 목적함수/가중 외 변수가 제거된다
      (A 에게 유리한 쪽으로 보수적)
    """
    out = {}
    p = make_params(mparams, "tweedie")
    fds = df["_fday"].to_numpy()
    for g in SOLO_GROUPS:
        fm = df[g].notna().to_numpy()
        X, y = df[feats], df[g]
        for fi, (tr, va, _) in enumerate(folds):
            trf = tr[fm[tr]]
            m, _ = fit_no_leak(mtype, p, X.iloc[trf], y.iloc[trf], fds[trf],
                               es_rounds=esr, mode="refit", seed=seed)
            imp = pd.Series(m.feature_importances_, index=feats)
            out[(g, fi)] = list(imp.sort_values(ascending=False).head(k).index)
    long = build_long(df, feats, JOINT_GROUPS)
    lfeats = [c for c in long.columns if not c.startswith("_")]
    fm = np.isfinite(long["_y"].to_numpy())
    Xl, yl = long[lfeats], long["_y"]
    lyr, lfd = long["_year"].to_numpy(), long["_fday"].to_numpy()
    for fi, (tr, va, _) in enumerate(make_folds(lyr, lfd, scheme="loyo")):
        trf = tr[fm[tr]]
        m, _ = fit_no_leak(mtype, p, Xl.iloc[trf], yl.iloc[trf], lfd[trf],
                           es_rounds=esr, mode="refit", seed=seed)
        imp = pd.Series(m.feature_importances_, index=lfeats)
        keep = list(imp.sort_values(ascending=False).head(k).index)
        for c in ["grp_0", "grp_1", "grp_2", "grp_is_vestas", "grp_n_turb", "grp_rotor_d"]:
            if c not in keep:
                keep.append(c)                    # 그룹 식별자는 항상 유지
        out[("joint", fi)] = keep
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step9")
    ap.add_argument("--top-k", type=int, default=None,
                    help="폴드별 top-k 가지치기 (v13 은 200). 미지정 시 전체 피처")
    ap.add_argument("--low-weight", type=float, default=0.2,
                    help="채점 제외행(actual<10%%cap) 가중치")
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
        tgt = df[f"{g}_cf"].to_numpy(float) * CREF
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * CREF)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * CREF)

    print(BAR); print("STEP 2/5  산식정합 가중 생성")
    for g in targets:
        df[f"_w_{g}"] = make_weight(answer[g].to_numpy(float),
                                    CAPACITY_KWH[g], low_w=args.low_weight)
        a = answer[g].to_numpy(float)
        fin = np.isfinite(a)
        sc = fin & (a >= 0.10 * CAPACITY_KWH[g])
        wv = df[f"_w_{g}"].to_numpy()[fin]
        print(f"  {g}: 라벨 {fin.sum():,}행  채점대상 {sc.sum():,}행 "
              f"({sc.sum()/max(fin.sum(),1)*100:.1f}%)  "
              f"가중 min {wv.min():.2f} / 중앙 {np.median(wv):.2f} / max {wv.max():.2f}")

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets] + [f"_w_{g}" for g in targets])
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    print(f"  행 {len(df):,}  피처 {len(feats)}개")

    mtype, mparams = cfg.get("model_type", "XGBoost"), cfg.get("model_params", {})
    esr, seed = cfg.get("early_stopping_rounds", 50), cfg["seed"]
    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
    folds = make_folds(yrs, fds, scheme="loyo")

    top_feats = {}
    if args.top_k:
        print(BAR); print(f"STEP 3/5  폴드별 top-{args.top_k} 선택 (조건 A 기준, 전 조건 공통)")
        top_feats = select_top_feats(df, feats, targets, folds, mtype, mparams,
                                     esr, seed, args.top_k)
        print(f"  완료 ({time.time()-t0:.0f}s)")
    else:
        print(BAR); print("STEP 3/5  가지치기 없음 (전체 피처로 가장 깨끗한 비교)")

    print(BAR); print("STEP 4/5  4개 조건 학습")
    oofs = {}
    for name, obj, use_w in CONDS:
        oof = run_condition(name, obj, use_w, df, feats, targets, folds,
                            mtype, mparams, esr, seed, scale, top_feats)
        oofs[name] = oof
        s = total_score(answer, oof, targets)
        print(f"  {name:12s} Total {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}"
              f"   ({time.time()-t0:.0f}s)")
        oof.to_csv(odir / f"oof_{name.split()[0]}.csv")

    print(BAR); print("STEP 5/5  판정")

    print("  [그룹별 raw 점수]")
    print(f"    {'조건':14s}" + "".join(f"{g.replace('kpx_',''):>14s}" for g in targets) + f"{'평균':>10s}")
    gsc = {}
    for n, o in oofs.items():
        ss = [group_score(answer[g].to_numpy(float), o[g].to_numpy(float), CAPACITY_KWH[g])[0]
              for g in targets]
        gsc[n] = ss
        print(f"    {n:14s}" + "".join(f"{v:14.4f}" for v in ss) + f"{np.mean(ss):10.4f}")

    print("\n  [연도별]")
    for n, o in oofs.items():
        score_by_year(df, answer, o, targets, label=n)

    print("\n  [요인 분해 — 이게 이 실험의 본체]")
    mA, mB = np.mean(gsc["A T+U(현행)"]), np.mean(gsc["B T+W"])
    mC, mD = np.mean(gsc["C L+U"]), np.mean(gsc["D L+W"])
    print(f"    목적함수 주효과 (L-T)  : {((mC+mD)-(mA+mB))/2:+.4f}")
    print(f"    가중     주효과 (W-U)  : {((mB+mD)-(mA+mC))/2:+.4f}")
    print(f"    상호작용               : {((mD-mC)-(mB-mA))/2:+.4f}")
    print("    (주효과가 노이즈 바닥 ±0.005 를 넘는지 + 아래 연도짝비교 부호가 일관한지)")

    print("\n  [연도짝비교 — vs A 현행]")
    base = oofs["A T+U(현행)"]
    for n, o in oofs.items():
        if n.startswith("A"):
            continue
        print(f"  --- {n} vs A ---")
        is_difference_real(df, answer, o, base, targets, name_a=n, name_b="A")

    print("\n  [그룹별 연도짝비교 — 총점보다 민감 (§3.7)]")
    yrs_u = [y for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]
    for n, o in oofs.items():
        if n.startswith("A"):
            continue
        print(f"    {n} - A")
        for g in targets:
            row = []
            for y in yrs_u:
                m = yrs == y
                sa = group_score(answer[g].to_numpy(float)[m], base[g].to_numpy(float)[m],
                                 CAPACITY_KWH[g])[0]
                sb = group_score(answer[g].to_numpy(float)[m], o[g].to_numpy(float)[m],
                                 CAPACITY_KWH[g])[0]
                row.append(np.nan if (np.isnan(sa) or np.isnan(sb)) else sb - sa)
            v = np.array(row, float); v = v[np.isfinite(v)]
            tag = ""
            if len(v) >= 2:
                tag = "  <- 부호일관" if (np.all(v > 0) or np.all(v < 0)) else ""
            print(f"      {g:14s}" + "".join(f"{x:+10.4f}" if np.isfinite(x) else f"{'-':>10s}"
                                             for x in row)
                  + f"  평균 {v.mean():+.4f}{tag}")

    best = max(oofs, key=lambda t: np.mean(gsc[t]))
    print(BAR); print(f"후처리 후 (A 현행 vs 최고 raw {best})")
    for n in dict.fromkeys(["A T+U(현행)", best]):
        print(f"  [{n}]")
        nested_postprocess_score(df, answer, oofs[n], targets, optimize_postprocessing,
                                 apply_postprocessing, mode="piecewise", verbose=True)

    json.dump({n: dict(zip(targets, map(float, s))) for n, s in gsc.items()},
              open(odir / "group_scores.json", "w"), indent=2)
    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")
    print("판정 규칙: 주효과 |값| > 0.005 이고 연도짝비교 부호가 3/3 일관일 때만 채택.")


if __name__ == "__main__":
    main()