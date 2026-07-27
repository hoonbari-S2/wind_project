"""
scripts/step12_conditional_post.py
==============================================================================
B-1 조건부 후처리 — 전역 단조 변환의 '강도' 를 행별 불확실성으로 변조한다

가설 (§3.13)
  현재 후처리는 전역 단조 변환이라 모든 행에 같은 매핑을 적용한다.
  그런데 밴드의 최적 행동은 행마다 다르다.
    - 불확실성이 6%*cap 보다 작은 행 : 밀면 밴드에 들어간다.        밀어야 이득
    - 불확실성이 그보다 큰 행         : 밀어도 못 들어간다. NMAE 만 버린다. 밀면 손해
  전역 변환은 둘을 구분하지 못하고 밴드를 살 수 없는 행에서도 밴드 값을 지불한다.

불확실성 대용치 — 재학습 0회
  원래는 폴드 모델 3개의 산포를 쓰려 했으나, OOF 에는 산포가 없다.
  LOYO 3년에서 어떤 연도를 제외한 모델은 하나뿐이고, 3개 평균은 test 에만 있다.
  적합할 데이터 자체가 없다.

  대신 양쪽에 다 존재하는 산포원을 쓴다.  s_i = |p13_i - p14_i| / cap
    - OOF        : saved_models/v13,v14 의 oof_preds.csv
    - test       : saved_models/v13,v14 의 raw_test_preds.csv
  단순 시드 노이즈가 아니라 '학습 문제가 다른 두 모델의 불일치' 이므로
  불확실성 신호로서 오히려 더 의미가 있다 (Tweedie=평균 계열 / MAE=중앙값 계열).

설계 — 2단, 기존 후처리를 엄격히 포함한다
  1단: 기존 optimize_postprocessing 그대로 (piecewise 7파라미터). 손대지 않는다.
  2단: final = raw + lam(q) * (post - raw),  lam(q) = lam_hi + (lam_lo - lam_hi)*q
       q = 그 행의 산포 분위(0=가장 확실, 1=가장 불확실). 파라미터 2개뿐.

  **lam_lo = lam_hi = 1 이면 현행 후처리와 정확히 같다.**
  즉 현행이 새 방법에 중첩(nested)되어 있어, 개선이 나오면 그것은 순수 추가분이다.
  파라미터도 2개만 늘어 §3.6 이 지적한 분산 문제를 최소화한다.

PART 1 — 기전 확인 (적합 없음). 여기서 막히면 PART 2 는 볼 필요 없다.
  산포 5분위별로 '후처리가 도움이 되는가' 를 직접 잰다.
  가설이 옳다면 Δ(후처리 − raw) 가 산포에 대해 **단조 감소**해야 하고,
  최상위 분위에서는 음수여야 한다. 그 패턴이 없으면 가설 기각이고 제출하지 않는다.

PART 2 — 조건부 후처리 적합 및 중첩 평가
  연도 하나를 빼고 1단·2단을 적합해 그 해에 적용한다. 현행(lam=1,1)과 짝비교.
  §3.6 규칙 5: 같은 OOF 위에서 후처리 방법만 바꾸는 짝비교라 모델 간 비교보다
  분산이 작다. 그래도 최종 판정은 LB.

실행
    python scripts/step12_conditional_post.py --config configs/config_v13.yaml
    python scripts/step12_conditional_post.py --config configs/config_v13.yaml --make-submission
==============================================================================
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import CAPACITY_KWH
from src.validation import quiet_warnings, add_time_keys, total_score, group_score
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
NQ = 5                                   # PART 1 분위 수
LAM_BOUNDS = (0.0, 1.3)


# ------------------------------------------------------------------ 산포
def dispersion(p_a, p_b, cap):
    """두 모델 예측의 불일치. 용량 정규화."""
    return np.abs(np.asarray(p_a, float) - np.asarray(p_b, float)) / cap


def quantile_map(s_fit):
    """
    적합셋의 산포 분포로 '분위 사상' 을 만든다. 평가셋에는 이 사상을 그대로 적용한다.
    평가셋 자신의 분포로 순위를 매기면 평가셋 전체를 본 것이 되므로 그렇게 하지 않는다.
    """
    v = np.sort(np.asarray(s_fit, float)[np.isfinite(s_fit)])
    if len(v) < 10:
        return lambda x: np.full(len(np.atleast_1d(x)), 0.5)
    qs = np.linspace(0.0, 1.0, len(v))
    return lambda x: np.interp(np.asarray(x, float), v, qs, left=0.0, right=1.0)


def lam_of_q(q, lam_lo, lam_hi):
    """q=0(가장 확실) -> lam_hi,  q=1(가장 불확실) -> lam_lo. 선형."""
    return lam_hi + (lam_lo - lam_hi) * np.clip(q, 0.0, 1.0)


def blend(raw, post, q, lam_lo, lam_hi, cap):
    lam = lam_of_q(q, lam_lo, lam_hi)
    return np.clip(raw + lam * (post - raw), 0.0, cap)


# ------------------------------------------------------------------ 2단 적합
def fit_lambda(actual, raw, post, q, cap, verbose=False):
    """
    lam_lo, lam_hi 두 개만 적합한다. 시작점 (1,1) = 현행.
    Nelder-Mead 는 x0 성분이 0 이면 0.00025 스텝을 쓰므로(§5.1 의 그 버그) 항상
    initial_simplex 를 명시한다. 여기선 스케일이 1 근처라 0.25 로 준다.
    """
    a = np.asarray(actual, float)
    ok = np.isfinite(a) & np.isfinite(raw) & np.isfinite(post) & np.isfinite(q)
    ok &= (a >= 0.10 * cap)                        # 채점 대상만으로 적합
    if ok.sum() < 100:
        return 1.0, 1.0
    a_, r_, p_, q_ = a[ok], raw[ok], post[ok], q[ok]

    def neg(x):
        lo, hi = np.clip(x, *LAM_BOUNDS)
        f = blend(r_, p_, q_, lo, hi, cap)
        return -group_score(a_, f, cap)[0]

    best, bx = np.inf, np.array([1.0, 1.0])
    for x0 in [np.array([1.0, 1.0]), np.array([0.5, 1.2]), np.array([0.2, 1.0])]:
        sim = np.vstack([x0, x0 + [0.25, 0.0], x0 + [0.0, 0.25]])
        try:
            r = minimize(neg, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=sim, xatol=1e-3, fatol=1e-6,
                                      maxiter=400, disp=False))
            if np.isfinite(r.fun) and r.fun < best:
                best, bx = r.fun, r.x
        except Exception:
            continue
    lo, hi = float(np.clip(bx[0], *LAM_BOUNDS)), float(np.clip(bx[1], *LAM_BOUNDS))
    if verbose:
        print(f"        lam_lo(불확실)={lo:.3f}  lam_hi(확실)={hi:.3f}")
    return lo, hi


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-base", default="./saved_models/v13", help="후처리를 적용할 기준 모델")
    ap.add_argument("--dir-alt", default="./saved_models/v14", help="산포 산출용 대조 모델")
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--out-name", default="submit_v15_condpost.csv")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    db, da = Path(args.dir_base), Path(args.dir_alt)

    tdir = Path(cfg["data_paths"]["train_dir"])
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()
    yrs = df["_year"].to_numpy()
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]

    ob = pd.read_csv(db / "oof_preds.csv", index_col=0).reindex(index=df.index, columns=targets).astype(float)
    oa = pd.read_csv(da / "oof_preds.csv", index_col=0).reindex(index=df.index, columns=targets).astype(float)

    print(BAR)
    print(f"기준 {db}  /  산포원 {da}")
    sb = total_score(answer, ob, targets)
    print(f"  기준 raw OOF {sb[0]:.4f}  (1-NMAE {sb[1]:.4f} / FICR {sb[2]:.4f})   연도 {years}")
    for g in targets:
        s = dispersion(ob[g], oa[g], CAPACITY_KWH[g])
        s = s[np.isfinite(s)]
        print(f"  {g}: 산포 중앙값 {np.median(s)*100:.2f}%cap  "
              f"90분위 {np.quantile(s,0.9)*100:.2f}%cap  (밴드 경계는 6%)")

    # ================================================================ PART 1
    print(BAR)
    print("PART 1  기전 확인 — 산포 분위별로 후처리가 도움이 되는가 (적합 없음)")
    print("  가설이 옳다면 Δ 가 산포에 대해 단조 감소하고 최상위 분위에서 음수여야 한다.\n")

    cells = {q: [] for q in range(NQ)}
    for y in years:
        te = (yrs == y)
        tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        post = apply_postprocessing(ob.loc[te].copy(), pp)
        for g in targets:
            cap = CAPACITY_KWH[g]
            s_fit = dispersion(ob.loc[tr, g], oa.loc[tr, g], cap)
            edges = np.nanquantile(s_fit[np.isfinite(s_fit)], np.linspace(0, 1, NQ + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            s_te = dispersion(ob.loc[te, g], oa.loc[te, g], cap)
            a = answer.loc[te, g].to_numpy(float)
            r = ob.loc[te, g].to_numpy(float)
            p = post[g].to_numpy(float)
            for qi in range(NQ):
                m = (s_te >= edges[qi]) & (s_te < edges[qi + 1]) & np.isfinite(a)
                if m.sum() < 100:
                    continue
                s_raw = group_score(a[m], r[m], cap)[0]
                s_pst = group_score(a[m], p[m], cap)[0]
                if np.isfinite(s_raw) and np.isfinite(s_pst):
                    cells[qi].append((s_pst - s_raw, m.sum()))

    print(f"  {'산포 분위':<12s}{'Δ(후처리−raw)':>16s}{'표준편차':>12s}{'양수/전체':>12s}{'평균 행수':>10s}")
    means = []
    for qi in range(NQ):
        v = np.array([c[0] for c in cells[qi]], float)
        n = np.array([c[1] for c in cells[qi]], float)
        if len(v) == 0:
            means.append(np.nan); continue
        means.append(v.mean())
        lab = f"Q{qi+1} {'(최확실)' if qi==0 else '(최불확실)' if qi==NQ-1 else ''}"
        print(f"  {lab:<12s}{v.mean():+16.4f}{v.std(ddof=1) if len(v)>1 else 0:12.4f}"
              f"{f'{(v>0).sum()}/{len(v)}':>12s}{n.mean():10.0f}")

    mv = np.array(means, float)
    fin = np.isfinite(mv)
    dec = bool(np.all(np.diff(mv[fin]) < 0)) if fin.sum() >= 3 else False
    span = float(np.nanmax(mv) - np.nanmin(mv))
    print(f"\n  단조 감소: {dec}   전체 폭 {span:.4f}   "
          f"최상위 분위 {mv[fin][-1]:+.4f}")
    if dec and mv[fin][-1] < 0:
        print("  => ✅ 기전 확인. 불확실한 행에서 후처리가 실제로 손해다.")
    elif span > 0.01 and mv[fin][-1] < mv[fin][0]:
        print("  => ⚠️ 방향은 맞으나 단조가 아니다. 약한 지지. PART 2 를 보되 기대를 낮춘다.")
    else:
        print("  => ❌ 기전 없음. 후처리 이득이 산포와 무관하다. 여기서 멈추고 제출하지 않는다.")

    # ================================================================ PART 2
    print(BAR)
    print("PART 2  조건부 후처리 적합 및 중첩 평가")
    print("  lam_lo=lam_hi=1 이면 현행과 동일하다. 적합 결과가 (1,1) 근처면 가설은 죽은 것이다.\n")

    rows = []
    for y in years:
        te = (yrs == y)
        tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        post_tr = apply_postprocessing(ob.loc[tr].copy(), pp)
        post_te = apply_postprocessing(ob.loc[te].copy(), pp)

        cur = ob.loc[te].copy()          # 현행 = lam 1
        new = ob.loc[te].copy()
        print(f"  [{y}]")
        for g in targets:
            cap = CAPACITY_KWH[g]
            s_tr = dispersion(ob.loc[tr, g], oa.loc[tr, g], cap)
            qmap = quantile_map(s_tr)
            q_tr = qmap(s_tr)
            q_te = qmap(dispersion(ob.loc[te, g], oa.loc[te, g], cap))
            lo, hi = fit_lambda(answer.loc[tr, g].to_numpy(float),
                                ob.loc[tr, g].to_numpy(float),
                                post_tr[g].to_numpy(float), q_tr, cap)
            cur[g] = post_te[g].to_numpy(float)
            new[g] = blend(ob.loc[te, g].to_numpy(float), post_te[g].to_numpy(float),
                           q_te, lo, hi, cap)
            gc = group_score(answer.loc[te, g].to_numpy(float), cur[g].to_numpy(float), cap)[0]
            gn = group_score(answer.loc[te, g].to_numpy(float), new[g].to_numpy(float), cap)[0]
            flag = "" if abs(lo - 1) + abs(hi - 1) > 0.10 else "   <- (1,1) 근처: 가설 무효"
            print(f"    {g:14s} lam_lo {lo:.3f} / lam_hi {hi:.3f}   "
                  f"현행 {gc:.4f} -> 조건부 {gn:.4f}  Δ {gn-gc:+.4f}{flag}")
        sc = total_score(answer.loc[te], cur, targets)[0]
        sn = total_score(answer.loc[te], new, targets)[0]
        rows.append((y, sc, sn))
        print(f"    총점            현행 {sc:.4f} -> 조건부 {sn:.4f}  Δ {sn-sc:+.4f}")

    d = np.array([r[2] - r[1] for r in rows], float)
    print(BAR)
    print("판정 (§3.6 규칙 5 — 같은 OOF 위 짝비교라 분산이 작다. 그래도 최종은 LB)")
    print(f"  현행 평균 {np.mean([r[1] for r in rows]):.4f}  "
          f"조건부 평균 {np.mean([r[2] for r in rows]):.4f}")
    print(f"  연도별 차이 [" + " ".join(f"{v:+.4f}" for v in d) + f"]  평균 {d.mean():+.4f}  "
          f"표준편차 {d.std(ddof=1) if len(d)>1 else 0:.4f}")
    sign_ok = np.all(d > 0) or np.all(d < 0)
    real = sign_ok and abs(d.mean()) > (d.std(ddof=1) if len(d) > 1 else np.inf)
    print(f"  부호일치 {sign_ok}  =>  {'✅ 신호' if real else '❌ 노이즈. 제출하지 않는다'}")

    # ================================================================ 제출
    if args.make_submission:
        print(BAR)
        tb, ta = db / "raw_test_preds.csv", da / "raw_test_preds.csv"
        if not (tb.exists() and ta.exists()):
            print(f"  ⚠ raw_test_preds.csv 가 없다 ({tb} / {ta}). "
                  f"각 config 로 main/inference.py 를 한 번씩 돌릴 것.")
        elif not real:
            print("  ⛔ PART 2 가 노이즈 판정이다. 제출 파일을 만들지 않는다.")
        else:
            pb, pa = pd.read_csv(tb), pd.read_csv(ta)
            assert len(pb) == len(pa), "두 예측 행 수가 다르다"
            # 적합은 가진 데이터 전부(3년)로 한다. 중첩은 절차의 정직한 평가치일 뿐이다.
            pp = optimize_postprocessing(answer, ob, mode="piecewise", verbose=False)
            post_full = apply_postprocessing(ob.copy(), pp)
            sub = pb.copy()
            post_test = apply_postprocessing(pb.copy(), pp)
            for g in targets:
                cap = CAPACITY_KWH[g]
                s_all = dispersion(ob[g], oa[g], cap)
                qmap = quantile_map(s_all)
                lo, hi = fit_lambda(answer[g].to_numpy(float), ob[g].to_numpy(float),
                                    post_full[g].to_numpy(float), qmap(s_all), cap)
                q_te = qmap(dispersion(pb[g], pa[g], cap))
                sub[g] = blend(pb[g].to_numpy(float), post_test[g].to_numpy(float),
                               q_te, lo, hi, cap)
                print(f"  {g}: lam_lo {lo:.3f} / lam_hi {hi:.3f}")
            sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            outdir = Path(cfg["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / args.out_name
            sub.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  🚀 제출 파일 저장: {out}")

    print(BAR); print(f"⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()