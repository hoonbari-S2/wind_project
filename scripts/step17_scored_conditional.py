"""
scripts/step17_scored_conditional.py
==============================================================================
밴드 트랙의 마지막 남은 축 — 후처리가 표현할 수 없는 변수로 저역을 조건화한다

세 번의 실패가 하나의 이유를 가리킨다
  step12  산포로 후처리 강도 변조        -> 교란. 폐기 (step15)
  step14  시각/계절 버킷 배율            -> m=1.000. 폐기
  step16  저역 매듭 증설 (7 -> 11 파라미터) -> 3/3 일관 '악화' (-0.0015)

  셋 다 **예측값 p 의 함수 공간 안에서만** 움직였다.
  그리고 step15 가 보여준 것: 후처리 이득의 52%가 L1(예측 최저 3분위)에서 나오는데,
  현행 격자의 [0, 0.15] 구간 사상은 0.17 -> 0.206 으로 사실상 **상수**다.
  step16 은 그 구간에 자유도를 5배 줬는데 오히려 나빠졌다.

  => L1 에서 최적 행동이 상수라는 것은, **그 구간에서 p 가 정보를 담고 있지 않다**는 뜻이다.
     (애초에 p 가 정보를 담았다면 그렇게 낮게 예측하고 틀리지 않았다)
     p 의 함수인 후처리는 L1 에서 원리적으로 더 짤 것이 없다. 밴드 트랙은 이 축에서 포화다.

남은 축 — 다른 타깃을 예측한다
  채점 필터가 a >= 0.10cap 이므로, 저역에서 점수를 최대화하는 점예측은
      E[a | x]            (현행 회귀가 주는 것)
  가 아니라
      E[a | a >= 0.10cap, x]   (채점될 조건 하의 기댓값)
  이다. §1.3 의 floor 우월성 증명이 정확히 이 이야기이고, 현행은 그 값을
  **전역 상수 0.17cap 하나**로 근사하고 있다. x 로 조건화하면 남는다.

  §3.8 의 따름정리("NWP 로 유도 가능한 것은 이미 모델이 갖고 있다")에 걸리지 않는다.
  스태킹은 같은 타깃을 재계산했지만 이건 **다른 타깃**이다. 같은 p 를 가진 두 행이
  서로 다른 E[a | a>=0.10cap, x] 를 가질 수 있고, 트리 회귀는 그 차이를 점추정 하나로
  압축해 버린다. 후처리는 p 의 함수라 원리적으로 되살릴 수 없다.
  §3.10 의 흡수 문제도 피한다 — 후처리가 아예 갖고 있지 않은 축이다.

--------------------------------------------------------------------------
PART 1  진단 (학습 없음) — L1 안에서 p 가 정말 정보가 없는가
  L1 행을 p 로 다시 5분할해 (a) 채점 비율 (b) 채점행 평균 실제값을 본다.
  p 가 정보를 담고 있다면 둘 다 p 에 대해 단조 증가해야 한다.
  평평하면 '저역에서 p 는 무정보' 가 확증되고 PART 2 의 전제가 선다.

PART 2  M_scored 학습 — 채점행(a >= 0.10cap)만으로 LOYO 학습
  타깃/피처/폴드는 v13 과 동일(정제 타깃, top-200, LOYO). 학습 표본만 채점행으로 제한한다.
  이것이 곧 Ê[a | a >= 0.10cap, x] 다.
  ⚠️ 단순화: v13 의 G1·G3 통합(long-format)은 쓰지 않고 그룹 독립으로 학습한다.
     이게 통과하면 통합을 얹어 재측정한다 (§3.7 에 따라 G3 가 가장 이득일 것).

PART 3  저역 대체 + 중첩 평가
      f = post                                   (p/cap >= TH)
      f = (1-lam)*post + lam*q                   (p/cap <  TH)
  lam = 0 이면 현행과 정확히 같다(중첩). TH 와 lam 을 적합셋에서만 고른다.

판정 (사전 등록 — 결과 보기 전 고정)
  1) PART 1 에서 L1 내 p 기울기가 평평해야 한다. 가파르면 전제가 틀렸으니 중단.
  2) 연도짝비교 부호 3/3 일치 ∧ |평균| > 표준편차  ∧  **부호가 양수**
     (step16 에서 '3/3 일관 악화' 를 겪었다. 방향을 명시하지 않은 것이 그때 규칙의 결함이었다)
  3) 그룹별 분해에서 최소 2개 그룹이 양수
  셋 다여야 제출 1회.

실행
    python scripts/step17_scored_conditional.py --config configs/config_v13.yaml
    python scripts/step17_scored_conditional.py --config configs/config_v13.yaml --make-submission
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
                            total_score, group_score, is_difference_real)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
CREF = 21600.0
TH_GRID = [0.08, 0.12, 0.16, 0.22, 0.30]
LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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
    ap.add_argument("--dir-base", default="./saved_models/v13")
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step17")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--out-name", default="submit_v15_scored.csv")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    sd = cfg["seed"]
    esr = cfg.get("early_stopping_rounds", 30)
    mtype = cfg.get("model_type", "XGBoost")
    mparams = cfg.get("model_params", {})
    db = Path(args.dir_base)
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    # ------------------------------------------------------------ 데이터
    print(BAR); print("STEP 1  데이터 + 피처 + 정제 타깃 (v13 과 동일 구성)")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = add_time_keys(build_full_feature_pipeline(
        labels.merge(w, on="forecast_kst_dtm", how="left")).replace([np.inf, -np.inf], np.nan))

    answer = df[targets].copy()                     # 원본 라벨 — 채점은 항상 이것으로
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
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]
    folds = make_folds(yrs, fds, scheme="loyo")

    ob = pd.read_csv(db / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=targets).astype(float)
    s0 = total_score(answer, ob, targets)
    print(f"  행 {len(df):,}  피처 {len(feats)}  연도 {years}")
    print(f"  기준 {db}  raw OOF {s0[0]:.4f}")

    # ============================================================ PART 1
    print(BAR)
    print("PART 1  진단 — L1(예측 최저 3분위) 안에서 p 가 정보를 담고 있는가 (학습 없음)")
    print("  p 가 정보를 담았다면 채점비율과 채점행 평균실제값이 p 에 단조 증가해야 한다.\n")
    flat_ok = []
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float); p = ob[g].to_numpy(float)
        fin = np.isfinite(a) & np.isfinite(p)
        e3 = np.nanquantile(p[fin], [0, 1/3])
        l1 = fin & (p <= np.nanquantile(p[fin], 1/3))
        q = np.nanquantile(p[l1], np.linspace(0, 1, 6)); q[0], q[-1] = -np.inf, np.inf
        rate, mean_a = [], []
        for k in range(5):
            m = l1 & (p >= q[k]) & (p < q[k + 1])
            if m.sum() < 100:
                rate.append(np.nan); mean_a.append(np.nan); continue
            sc = m & (a >= 0.10 * cap)
            rate.append(sc.sum() / m.sum() * 100)
            mean_a.append(np.mean(a[sc]) / cap * 100 if sc.sum() >= 30 else np.nan)
        print(f"  [{g.replace('kpx_','')}]  L1 상한 p/cap = {np.nanquantile(p[fin],1/3)/cap:.3f}")
        print("    " + f"{'p 5분위':<12s}" + "".join(f"{'Q'+str(k+1):>10s}" for k in range(5)))
        print("    " + f"{'채점비율%':<12s}" + "".join(f"{v:10.1f}" if np.isfinite(v) else f"{'-':>10s}" for v in rate))
        print("    " + f"{'평균실제%cap':<12s}" + "".join(f"{v:10.1f}" if np.isfinite(v) else f"{'-':>10s}" for v in mean_a))
        v = np.array(mean_a, float); v = v[np.isfinite(v)]
        span = (v.max() - v.min()) if len(v) else np.nan
        flat_ok.append(span < 6.0)
        print(f"    평균실제값 전체 폭 {span:.1f}%cap"
              + ("   <- 평평. p 무정보 확인" if span < 6.0 else "   <- 가파르다. 전제 재검토"))
    print()
    if not any(flat_ok):
        print("  => ❌ 전 그룹에서 p 가 정보를 담고 있다. 이 실험의 전제가 틀렸다. 중단 권장.")
    else:
        print(f"  => ✅ {sum(flat_ok)}/{len(targets)} 그룹에서 저역 p 가 무정보. PART 2 로 간다.")

    # ============================================================ PART 2
    print(BAR)
    print("PART 2  M_scored 학습 — 채점행(원본 라벨 >= 0.10cap)만으로 LOYO")
    print("  이것이 Ê[a | a>=0.10cap, x] 다. 폴드별 top-200 은 v13 과 동일 절차.\n")
    qo = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    for g in targets:
        cap = CAPACITY_KWH[g]
        fit_m = (df[g].notna().to_numpy()
                 & np.isfinite(answer[g].to_numpy(float))
                 & (answer[g].to_numpy(float) >= 0.10 * cap))
        pred = np.full(len(df), np.nan)
        for tr, va, name in folds:
            trf = tr[fit_m[tr]]
            if len(trf) < 300:
                continue
            m0, _ = fit_no_leak(mtype, mparams, df[feats].iloc[trf], df[g].iloc[trf],
                                fds[trf], es_rounds=esr, mode="refit", seed=sd)
            imp = pd.Series(m0.feature_importances_, index=feats)
            cols = list(imp.sort_values(ascending=False).head(args.top_k).index)
            m, _ = fit_no_leak(mtype, mparams, df[cols].iloc[trf], df[g].iloc[trf], fds[trf],
                               es_rounds=esr, mode="refit", seed=sd)
            pred[va] = np.clip(m.predict(df[cols].iloc[va]), 0, 1.15 * CREF)
        qo[g] = np.clip(pred * scale[g], 0, cap)
        print(f"  {g}: 학습 {int(fit_m.sum()):,}행  ({time.time()-t0:.0f}s)")
    qo.to_csv(odir / "oof_scored.csv")

    # ============================================================ PART 3
    print(BAR)
    print("PART 3  저역 대체 + LOYO 중첩 평가  (lam=0 이면 현행과 동일)")
    print("  TH, lam 은 적합셋에서만 고른다. 평가연도는 보지 않는다.\n")

    def blend(post, p, q, cap, th, lam):
        low = (p / cap) < th
        out = post.copy()
        out[low] = (1 - lam) * post[low] + lam * q[low]
        return np.clip(out, 0, cap)

    cur = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    new = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    chosen = {}
    for y in years:
        te = (yrs == y); tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        post_tr = apply_postprocessing(ob.loc[tr].copy(), pp)
        post_te = apply_postprocessing(ob.loc[te].copy(), pp)
        print(f"  [{y}]")
        for g in targets:
            cap = CAPACITY_KWH[g]
            a_tr = answer.loc[tr, g].to_numpy(float)
            p_tr = ob.loc[tr, g].to_numpy(float); q_tr = qo.loc[tr, g].to_numpy(float)
            po_tr = post_tr[g].to_numpy(float)
            ok = np.isfinite(a_tr) & np.isfinite(p_tr) & np.isfinite(q_tr) & np.isfinite(po_tr)
            best, bth, blam = -np.inf, TH_GRID[0], 0.0
            for th in TH_GRID:
                for lam in LAM_GRID:
                    f = blend(po_tr[ok], p_tr[ok], q_tr[ok], cap, th, lam)
                    s = group_score(a_tr[ok], f, cap)[0]
                    if np.isfinite(s) and s > best:
                        best, bth, blam = s, th, lam
            chosen[(y, g)] = (bth, blam)
            p_te = ob.loc[te, g].to_numpy(float); q_te = qo.loc[te, g].to_numpy(float)
            po_te = post_te[g].to_numpy(float)
            q_te = np.where(np.isfinite(q_te), q_te, po_te)     # q 결측이면 현행 유지
            cur.loc[te, g] = po_te
            new.loc[te, g] = blend(po_te, p_te, q_te, cap, bth, blam)
            a_te = answer.loc[te, g].to_numpy(float)
            gc = group_score(a_te, po_te, cap)[0]
            gn = group_score(a_te, new.loc[te, g].to_numpy(float), cap)[0]
            flag = "   <- lam=0: 가설 무효" if blam == 0.0 else ""
            print(f"    {g:14s} TH {bth:.2f} / lam {blam:.1f}   "
                  f"{gc:.4f} -> {gn:.4f}  Δ {gn-gc:+.4f}{flag}")
        sc = total_score(answer.loc[te], cur.loc[te], targets)[0]
        sn = total_score(answer.loc[te], new.loc[te], targets)[0]
        print(f"    총점            {sc:.4f} -> {sn:.4f}  Δ {sn-sc:+.4f}")

    print(BAR); print("판정 (사전 등록: 3/3 부호일치 ∧ |평균|>표준편차 ∧ 부호 양수 ∧ 그룹 2개 이상 양수)")
    res = is_difference_real(df, answer, new, cur, targets, name_a="조건부저역", name_b="현행")
    print("\n  [그룹별]")
    gpos = 0
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float)
        gc = group_score(a, cur[g].to_numpy(float), cap)[0]
        gn = group_score(a, new[g].to_numpy(float), cap)[0]
        gpos += int(gn > gc)
        print(f"    {g:14s} {gc:.4f} -> {gn:.4f}  Δ {gn-gc:+.4f}")
    ok_all = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2)
    print(f"\n  부호양수 {bool(res and res['mean']>0)}  양수그룹 {gpos}/{len(targets)}"
          f"  =>  {'✅ 제출 1회' if ok_all else '❌ 제출하지 않는다'}")

    if args.make_submission:
        print(BAR)
        tb = db / "raw_test_preds.csv"
        if not tb.exists():
            print(f"  ⚠ {tb} 없음. main/inference.py 를 먼저 돌릴 것.")
        elif not ok_all:
            print("  ⛔ 판정 미달. 제출 파일을 만들지 않는다.")
        else:
            print("  ⚠ test 예측에는 M_scored 를 3년 전체로 재학습해 적용해야 한다.")
            print("    (이 스크립트는 OOF 판정까지만 한다 — 통과했으니 재학습 경로를 붙일 것)")

    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()