"""
scripts/step14_covariate_post.py
==============================================================================
B-1' 공변량 조건부 후처리 — 남은 '구조적' 오차를 찾는다

step12 의 교훈
  조건화 변수를 '불확실성' 으로 잡은 것이 틀렸다. 후처리는 밴드로 밀어넣는 장치가 아니라
  **구조적 편향을 걷어내는 장치**이고, 불확실성은 무작위 오차의 크기일 뿐이다.

프리미엄이 생기는 조건 (합성 실험으로 확인)
  분산만 1/k 로 줄인 오차는 FICR 을 크게 올리지만 **프리미엄은 0에서 안 움직인다.**
  반정규 함의 FICR 이 그 NMAE 개선을 이미 반영하기 때문이다.
    N=1 프리미엄 -0.0005 / N=3 +0.0007 / N=8 -0.0000
  즉,
    무작위 오차를 지우면      -> 정확도 레버
    구조적 편향을 지우면      -> 프리미엄 레버 (오차분포가 가우시안보다 뾰족해진다)

  현재 후처리는 **예측값의 함수일 뿐**이라 예측 수준에 따른 편향만 걷는다.
  시각·계절에 따른 편향이 남아 있다면 그것이 아직 안 캔 프리미엄이다.

PART 1 — 진단 (적합 없음). 여기서 편향이 0 이면 그냥 끝이다.
  현행 후처리를 통과한 뒤에도 버킷별로 부호 있는 평균오차가 남는가.
  잔여 편향이 밴드(6%cap) 대비 의미 있는 크기여야 캘 것이 있다.

PART 2 — 버킷별 배율 m_b 를 현행 후처리 위에 얹고 중첩 평가
  final = post * m_b.  m_b = 1 이면 현행과 정확히 같다 (중첩 구조 유지).
  연도 하나를 빼고 적합해 그 해에 적용. §3.6 규칙 5.

학습 없음. train_labels.csv 와 저장된 OOF 만 읽는다.

실행
    python scripts/step14_covariate_post.py --config configs/config_v13.yaml
    python scripts/step14_covariate_post.py --config configs/config_v13.yaml --bucket season
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
M_BOUNDS = (0.80, 1.20)


def make_buckets(t, kind):
    """t: pandas datetime Series -> (라벨 배열, 이름 리스트)"""
    h, mo = t.dt.hour.to_numpy(), t.dt.month.to_numpy()
    if kind == "hour4":
        b = np.digitize(h, [6, 12, 18])
        return b, ["00-05", "06-11", "12-17", "18-23"]
    if kind == "hour6":
        b = np.digitize(h, [4, 8, 12, 16, 20])
        return b, ["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"]
    if kind == "season":
        s = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        return np.array([s[m] for m in mo]), ["DJF", "MAM", "JJA", "SON"]
    if kind == "month":
        return mo - 1, [f"{m}월" for m in range(1, 13)]
    if kind == "daynight":
        return (np.digitize(h, [7, 19]) == 1).astype(int), ["야간", "주간"]
    raise ValueError(kind)


def fit_multipliers(actual, post, b, nb, cap):
    """
    버킷별 배율. 시작점 전부 1.0 (= 현행).
    Nelder-Mead 는 x0 이 0 이 아니면 ×1.05 로 시뮬렉스를 만드는데(§5.1 참조)
    여기선 1 근처라 큰 문제는 없지만, 재현성을 위해 명시한다.
    """
    a = np.asarray(actual, float)
    ok = np.isfinite(a) & np.isfinite(post) & (a >= 0.10 * cap)
    if ok.sum() < 200:
        return np.ones(nb)
    a_, p_, b_ = a[ok], post[ok], b[ok]

    def neg(x):
        m = np.clip(x, *M_BOUNDS)[b_]
        return -group_score(a_, np.clip(p_ * m, 0, cap), cap)[0]

    x0 = np.ones(nb)
    sim = np.vstack([x0] + [x0 + np.eye(nb)[i] * 0.05 for i in range(nb)])
    try:
        r = minimize(neg, x0, method="Nelder-Mead",
                     options=dict(initial_simplex=sim, xatol=1e-4, fatol=1e-7,
                                  maxiter=200 * nb, disp=False))
        return np.clip(r.x, *M_BOUNDS) if np.all(np.isfinite(r.x)) else x0
    except Exception:
        return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-base", default="./saved_models/v13")
    ap.add_argument("--bucket", default="hour4",
                    choices=["hour4", "hour6", "season", "month", "daynight"])
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--out-name", default="submit_v15_covpost.csv")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    db = Path(args.dir_base)

    tdir = Path(cfg["data_paths"]["train_dir"])
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()
    yrs = df["_year"].to_numpy()
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]
    tt = pd.to_datetime(df["forecast_kst_dtm"])
    ob = pd.read_csv(db / "oof_preds.csv", index_col=0).reindex(index=df.index, columns=targets).astype(float)

    print(BAR)
    s0 = total_score(answer, ob, targets)
    print(f"기준 {db}   raw OOF {s0[0]:.4f} (1-NMAE {s0[1]:.4f} / FICR {s0[2]:.4f})   연도 {years}")

    for kind in (["hour4", "hour6", "season", "month", "daynight"]
                 if args.bucket == "hour4" else [args.bucket]):
        pass    # 진단은 아래에서 여러 축을 한꺼번에 본다

    # ================================================================ PART 1
    print(BAR)
    print("PART 1  진단 — 현행 후처리를 통과한 뒤에도 남는 '부호 있는' 편향 (적합 없음)")
    print("  단위는 %cap. 밴드가 ±6% 이므로 잔여 편향이 1%cap 을 넘으면 캘 것이 있다.")
    print("  (예측값 축은 후처리가 이미 처리하므로 0 에 가까워야 정상 — 대조군)\n")

    # 연도 하나를 빼고 후처리를 적합해 그 해에 적용한 예측을 모은다
    post_all = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    for y in years:
        te = (yrs == y); tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        post_all.loc[te, targets] = apply_postprocessing(ob.loc[te].copy(), pp)[targets].values

    for kind in ["hour6", "season", "daynight"]:
        b, names = make_buckets(tt, kind)
        print(f"  [{kind}]")
        hdr = "    " + f"{'그룹':<14s}" + "".join(f"{n:>10s}" for n in names)
        print(hdr)
        for g in targets:
            cap = CAPACITY_KWH[g]
            a = answer[g].to_numpy(float); p = post_all[g].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(p) & (a >= 0.10 * cap)
            row = []
            for k in range(len(names)):
                m = ok & (b == k)
                row.append(np.mean(p[m] - a[m]) / cap * 100 if m.sum() >= 200 else np.nan)
            print("    " + f"{g.replace('kpx_',''):<14s}"
                  + "".join(f"{v:+10.2f}" if np.isfinite(v) else f"{'-':>10s}" for v in row))
        # 대조군: 예측 수준 5분위
        if kind == "hour6":
            print("    " + f"{'(대조) 예측5분위':<14s}", end="")
            g = targets[0]; cap = CAPACITY_KWH[g]
            a = answer[g].to_numpy(float); p = post_all[g].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(p) & (a >= 0.10 * cap)
            edges = np.nanquantile(p[ok], np.linspace(0, 1, 6)); edges[0], edges[-1] = -np.inf, np.inf
            for k in range(5):
                m = ok & (p >= edges[k]) & (p < edges[k + 1])
                v = np.mean(p[m] - a[m]) / cap * 100 if m.sum() >= 200 else np.nan
                print(f"{v:+10.2f}" if np.isfinite(v) else f"{'-':>10s}", end="")
            print()
        print()

    b, names = make_buckets(tt, args.bucket)
    nb = len(names)
    bias = []
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float); p = post_all[g].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(p) & (a >= 0.10 * cap)
        for k in range(nb):
            m = ok & (b == k)
            if m.sum() >= 200:
                bias.append(abs(np.mean(p[m] - a[m]) / cap * 100))
    mx = max(bias) if bias else 0.0
    print(f"  선택 축 [{args.bucket}] 최대 |잔여 편향| = {mx:.2f}%cap")
    if mx >= 1.0:
        print("  => ✅ 구조적 편향이 남아 있다. PART 2 로 간다.")
    elif mx >= 0.5:
        print("  => ⚠️ 편향이 작다. PART 2 이득도 작을 것이다.")
    else:
        print("  => ❌ 잔여 편향이 사실상 없다. 이 축에는 캘 프리미엄이 없다.")

    # ================================================================ PART 2
    print(BAR)
    print(f"PART 2  버킷별 배율 적합 및 중첩 평가 (축: {args.bucket}, 버킷 {nb}개)")
    print("  m_b = 1 이면 현행과 정확히 같다. 적합 결과가 전부 1 근처면 가설은 죽은 것이다.\n")

    rows = []
    for y in years:
        te = (yrs == y); tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        post_tr = apply_postprocessing(ob.loc[tr].copy(), pp)
        post_te = apply_postprocessing(ob.loc[te].copy(), pp)
        cur, new = ob.loc[te].copy(), ob.loc[te].copy()
        print(f"  [{y}]")
        for g in targets:
            cap = CAPACITY_KWH[g]
            m = fit_multipliers(answer.loc[tr, g].to_numpy(float),
                                post_tr[g].to_numpy(float), b[tr], nb, cap)
            cur[g] = post_te[g].to_numpy(float)
            new[g] = np.clip(post_te[g].to_numpy(float) * m[b[te]], 0, cap)
            gc = group_score(answer.loc[te, g].to_numpy(float), cur[g].to_numpy(float), cap)[0]
            gn = group_score(answer.loc[te, g].to_numpy(float), new[g].to_numpy(float), cap)[0]
            flat = "   <- 전부 1 근처: 가설 무효" if np.max(np.abs(m - 1)) < 0.01 else ""
            print(f"    {g:14s} m=[" + " ".join(f"{v:.3f}" for v in m) + "]"
                  + f"  {gc:.4f} -> {gn:.4f}  Δ {gn-gc:+.4f}{flat}")
        sc = total_score(answer.loc[te], cur, targets)[0]
        sn = total_score(answer.loc[te], new, targets)[0]
        rows.append((y, sc, sn))
        print(f"    총점            {sc:.4f} -> {sn:.4f}  Δ {sn-sc:+.4f}")

    d = np.array([r[2] - r[1] for r in rows], float)
    print(BAR); print("판정 (§3.6 규칙 5 — 같은 OOF 위 짝비교. 최종은 LB)")
    print(f"  현행 평균 {np.mean([r[1] for r in rows]):.4f}  "
          f"공변량 평균 {np.mean([r[2] for r in rows]):.4f}")
    print(f"  연도별 차이 [" + " ".join(f"{v:+.4f}" for v in d) + f"]  평균 {d.mean():+.4f}  "
          f"표준편차 {d.std(ddof=1) if len(d)>1 else 0:.4f}")
    sign_ok = bool(np.all(d > 0) or np.all(d < 0))
    real = sign_ok and abs(d.mean()) > (d.std(ddof=1) if len(d) > 1 else np.inf)
    print(f"  부호일치 {sign_ok}  =>  {'✅ 신호' if real else '❌ 노이즈. 제출하지 않는다'}")

    if args.make_submission:
        print(BAR)
        tb = db / "raw_test_preds.csv"
        if not tb.exists():
            print(f"  ⚠ {tb} 없음. main/inference.py 를 먼저 돌릴 것.")
        elif not real:
            print("  ⛔ 노이즈 판정이다. 제출 파일을 만들지 않는다.")
        else:
            pb = pd.read_csv(tb)
            pp = optimize_postprocessing(answer, ob, mode="piecewise", verbose=False)
            post_full = apply_postprocessing(ob.copy(), pp)
            sub = apply_postprocessing(pb.copy(), pp)
            bt, _ = make_buckets(pd.to_datetime(pb["forecast_kst_dtm"]), args.bucket)
            for g in targets:
                cap = CAPACITY_KWH[g]
                m = fit_multipliers(answer[g].to_numpy(float), post_full[g].to_numpy(float),
                                    b, nb, cap)
                sub[g] = np.clip(sub[g].to_numpy(float) * m[bt], 0, cap)
                print(f"  {g}: m=[" + " ".join(f"{v:.3f}" for v in m) + "]")
            sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            outdir = Path(cfg["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / args.out_name
            sub.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  🚀 제출 파일 저장: {out}")

    print(BAR); print(f"⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()