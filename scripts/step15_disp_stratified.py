"""
scripts/step15_disp_stratified.py
==============================================================================
B-1 재심 + step14 오진 기전 확인 — 학습 0회, 저장된 OOF 두 개만 읽는다

--------------------------------------------------------------------------
PART 1  step12 의 '반대 방향' 이 실재인가, 예측 수준 교란인가
--------------------------------------------------------------------------
step12 가 실제로 말한 것 (기억할 것: null 이 아니라 부호가 반대였다)

    가설   : 산포가 크면 후처리가 손해다 -> Δ 가 산포에 단조 '감소', 최상위 음수
    실측   : Δ = +0.0126 / +0.0096 / +0.0138 / +0.0278 / +0.0278   -> 단조 '증가'
    PART 2 : lam_lo(불확실) 1.05~1.30 / lam_hi(확실) 0.61~0.83     -> 6/6 같은 방향

  두 파트가 같은 말을 한다. **불확실한 행일수록 후처리가 더 이득이다.**
  가설의 반증이지 측정 실패가 아니다. 그런데 총점 Δ 는 노이즈였다(+0.0021/+0.0026/-0.0016).
  '방향은 6/6 일관, 크기는 노이즈' 는 보통 교란변수가 있을 때 나오는 모양이다.

교란 용의자
  s = |p13 - p14| / cap 은 불확실성 대용치이지만 **예측 수준의 대용치이기도 하다.**
  두 모델의 불일치는 저출력에서 작고 중·고출력에서 크다.
  한편 piecewise 후처리는 오직 예측 수준의 함수이고, 이득이 큰 구간
  (0.17cap floor / 고출력 하향수축)도 수준으로 정해진다.
  => '산포 5분위' 가 사실은 '예측 수준 5분위' 를 다시 그린 것일 수 있다.

  교란이면 : 예측 수준을 고정하면 산포 기울기가 사라진다.
  실재면   : 수준을 고정해도 기울기가 남는다. (부호가 어느 쪽이든 그것이 답이다)

  주의 — 기울기가 남더라도 부호는 **양(+)** 일 가능성이 높고, 그러면 원래 B-1
  가설(불확실한 행은 밀지 마라)은 죽고 그 **반대**가 산다. 그 경우 lam 을
  [0,1] 로 묶으면 안 된다는 뜻이 되므로 설계가 바뀐다. 부호를 먼저 확정한다.

설계
  연도 하나를 빼고 piecewise 후처리를 적합해 그 해에 적용 (step12 와 동일 절차).
  각 (연도, 그룹) 에서
    1) 예측 수준 NL 분할 — 경계는 **적합셋 분위로 고정**. 평가셋 분포를 보지 않는다.
    2) 각 수준 칸 '안에서' 다시 산포 NS 분할 — 경계도 적합셋의 같은 수준 칸에서.
  칸마다 Δ = score(post) - score(raw). NL x NS 표로 낸다.
  대조표로 수준 미고정 산포 분할(= step12 재현)도 같이 낸다.

판정 (사전 등록 — 결과 보기 전에 고정)
  기울기 = Δ(최고산포 칸) - Δ(최저산포 칸)
  A. 수준 고정 후 NL 개 수준 칸 전부에서 기울기 부호가 같고,
     |평균 기울기| >= 0.5 x |수준 미고정 기울기|
        -> 산포 효과 실재. B-1 을 '부호를 뒤집어' 재설계한다.
  B. 부호가 갈리거나 크기가 절반 미만
        -> 교란. B-1(§8.2) 최종 폐기. 산포 축을 더 파지 않는다.

--------------------------------------------------------------------------
PART 2  step14 PART 1 은 왜 거짓 ✅ 를 띄웠는가 — 기전 확인
--------------------------------------------------------------------------
step14 는 '후처리 후에도 평균 부호편향 +3~+9%cap 이 남는다' 를 근거로
"구조적 편향이 남아 있다 => PART 2 로 간다" 를 띄웠다. 그러나

  * piecewise 가 찾은 절점은 0 -> 0.17cap 이다 (§5.1). 즉 저예측 행은 17%cap 으로 올린다.
  * 채점 필터는 a >= 0.10cap 이다.
  * 따라서 a 가 0.10~0.17cap 인 행에서는 **정의상** p - a > 0 이다.
  * 그리고 §1.3 이 증명했듯 그 floor 는 점수 최적 행동이다.

  => 그 편향은 '아직 안 캔 프리미엄' 이 아니라 '이미 캔 것이 남긴 흔적' 이다.
     step14 PART 2 가 m=1.000 을 뱉은 것과 정확히 일치한다.

  확인 방법: 편향을 **실제값 구간별로** 쪼갠다. 기전이 맞으면
    - a in [0.10, 0.20)cap 에서 편향이 압도적으로 크고
    - 고출력 구간에서 0 에 가깝거나 부호가 뒤집혀야 한다.
  step14 처럼 전 구간을 뭉뚱그린 평균은 저출력 칸 하나에 끌려다닌다.

  기전이 확인되면 §3.13 / step14 의 진단 기준('잔여 편향 1%cap 이면 캘 것 있다')을
  폐기하고, 편향이 아니라 **밴드 적중률**을 진단 지표로 바꾼다.

실행
    python scripts/step15_disp_stratified.py --config configs/config_v13.yaml
    python scripts/step15_disp_stratified.py --config configs/config_v13.yaml --n-level 4
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
from src.utils import CAPACITY_KWH
from src.validation import quiet_warnings, add_time_keys, total_score, group_score
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
MIN_CELL = 100                      # 칸 최소 행수 (채점 필터 전)
A_EDGES = [0.10, 0.20, 0.35, 0.60, 1.10]        # PART 2 실제값 구간 (cap 정규화)


def q_edges(v, n):
    """적합셋 분위로 경계를 만든다. 양 끝은 무한대로 열어 평가셋 꼬리를 흡수한다."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < n * 20:
        return None
    e = np.nanquantile(v, np.linspace(0.0, 1.0, n + 1))
    e = np.unique(e)
    if len(e) < n + 1:
        return None
    e[0], e[-1] = -np.inf, np.inf
    return e


def binof(v, e):
    return np.clip(np.digitize(np.asarray(v, float), e[1:-1]), 0, len(e) - 2)


def delta_score(a, raw, post, cap, m):
    """칸 m 에서 후처리가 준 점수 차. 채점 대상(a>=10%cap)은 group_score 가 거른다."""
    if m.sum() < MIN_CELL:
        return np.nan
    s_r = group_score(a[m], raw[m], cap)[0]
    s_p = group_score(a[m], post[m], cap)[0]
    if not (np.isfinite(s_r) and np.isfinite(s_p)):
        return np.nan
    return s_p - s_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-base", default="./saved_models/v13")
    ap.add_argument("--dir-alt", default="./saved_models/v14", help="산포 산출용 대조 모델")
    ap.add_argument("--n-level", type=int, default=3)
    ap.add_argument("--n-disp", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    NL, NS = args.n_level, args.n_disp
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

    ob = pd.read_csv(db / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=targets).astype(float)
    oa = pd.read_csv(da / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=targets).astype(float)

    print(BAR)
    s0 = total_score(answer, ob, targets)
    print(f"기준 {db}  /  산포원 {da}")
    print(f"  raw OOF {s0[0]:.4f} (1-NMAE {s0[1]:.4f} / FICR {s0[2]:.4f})   연도 {years}")
    print(f"  격자: 예측수준 {NL} x 산포 {NS}   칸 최소 {MIN_CELL}행")

    # ---- 산포와 예측 수준의 상관 (교란 가설의 직접 증거) -------------------
    print(f"\n  [교란 사전 점검] 산포 s 와 예측 수준 p 의 스피어만 상관")
    for g in targets:
        cap = CAPACITY_KWH[g]
        p = ob[g].to_numpy(float)
        s = np.abs(p - oa[g].to_numpy(float)) / cap
        ok = np.isfinite(p) & np.isfinite(s)
        r = pd.Series(p[ok]).corr(pd.Series(s[ok]), method="spearman")
        print(f"    {g.replace('kpx_',''):<10s} rho = {r:+.3f}"
              + ("   <- 강한 교란. PART 1 없이는 step12 를 해석할 수 없다" if abs(r) > 0.5 else ""))

    # ======================================================== PART 1
    print(BAR)
    print("PART 1  예측 수준을 고정하고 산포 효과를 다시 본다")
    print("  각 칸의 값은 Δ = score(후처리) - score(raw). (연도x그룹) 평균이다.\n")

    cells = {(k, j): [] for k in range(NL) for j in range(NS)}
    ctrl = {j: [] for j in range(NS)}                 # 수준 미고정 대조 (step12 재현)
    post_all = pd.DataFrame(index=df.index, columns=targets, dtype=float)

    for y in years:
        te = (yrs == y)
        tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        post_te = apply_postprocessing(ob.loc[te].copy(), pp)
        post_all.loc[te, targets] = post_te[targets].values

        for g in targets:
            cap = CAPACITY_KWH[g]
            p_tr = ob.loc[tr, g].to_numpy(float)
            s_tr = np.abs(p_tr - oa.loc[tr, g].to_numpy(float)) / cap
            l_tr = p_tr / cap

            a = answer.loc[te, g].to_numpy(float)
            r_ = ob.loc[te, g].to_numpy(float)
            p_ = post_te[g].to_numpy(float)
            s_te = np.abs(r_ - oa.loc[te, g].to_numpy(float)) / cap
            l_te = r_ / cap
            fin = np.isfinite(a) & np.isfinite(r_) & np.isfinite(p_) & np.isfinite(s_te)

            # --- 대조: 수준 미고정 산포 분할
            e_all = q_edges(s_tr, NS)
            if e_all is not None:
                b_all = binof(s_te, e_all)
                for j in range(NS):
                    d = delta_score(a, r_, p_, cap, fin & (b_all == j))
                    if np.isfinite(d):
                        ctrl[j].append(d)

            # --- 본 실험: 수준 고정 후 산포 분할
            le = q_edges(l_tr, NL)
            if le is None:
                continue
            lb_tr, lb_te = binof(l_tr, le), binof(l_te, le)
            for k in range(NL):
                se = q_edges(s_tr[lb_tr == k], NS)
                if se is None:
                    continue
                sb = binof(s_te, se)
                for j in range(NS):
                    d = delta_score(a, r_, p_, cap, fin & (lb_te == k) & (sb == j))
                    if np.isfinite(d):
                        cells[(k, j)].append(d)

    def cellmean(v):
        return np.mean(v) if len(v) else np.nan

    hdr = "    " + f"{'예측수준':<12s}" + "".join(
        f"{('S'+str(j+1)+(' 최저' if j==0 else ' 최고' if j==NS-1 else '')):>12s}" for j in range(NS)) \
        + f"{'기울기':>12s}{'n칸':>6s}"
    print(hdr)
    slopes = []
    for k in range(NL):
        vals = [cellmean(cells[(k, j)]) for j in range(NS)]
        sl = vals[-1] - vals[0]
        slopes.append(sl)
        lab = f"L{k+1}" + (" 최저" if k == 0 else " 최고" if k == NL - 1 else "")
        print("    " + f"{lab:<12s}"
              + "".join(f"{v:+12.4f}" if np.isfinite(v) else f"{'-':>12s}" for v in vals)
              + (f"{sl:+12.4f}" if np.isfinite(sl) else f"{'-':>12s}")
              + f"{len(cells[(k,0)]):6d}")

    cvals = [cellmean(ctrl[j]) for j in range(NS)]
    cslope = cvals[-1] - cvals[0]
    print("    " + "-" * (12 + 12 * NS + 18))
    print("    " + f"{'(대조) 미고정':<12s}"
          + "".join(f"{v:+12.4f}" if np.isfinite(v) else f"{'-':>12s}" for v in cvals)
          + f"{cslope:+12.4f}{len(ctrl[0]):6d}")

    sl = np.array(slopes, float)
    fin_sl = sl[np.isfinite(sl)]
    sign_ok = bool(len(fin_sl) == NL and (np.all(fin_sl > 0) or np.all(fin_sl < 0)))
    ratio = abs(np.mean(fin_sl)) / abs(cslope) if np.isfinite(cslope) and abs(cslope) > 1e-12 else np.nan
    print(f"\n  수준 고정 기울기 [" + " ".join(f"{v:+.4f}" for v in sl) + f"]  평균 {np.mean(fin_sl):+.4f}")
    print(f"  수준 미고정 기울기 {cslope:+.4f}   잔존 비율 {ratio:.2f}")
    print(f"  부호일치 {sign_ok}")
    if sign_ok and np.isfinite(ratio) and ratio >= 0.5:
        d = "양(+) — '불확실할수록 더 밀어라'" if np.mean(fin_sl) > 0 else "음(-) — 원래 B-1 가설대로"
        print(f"  => ✅ A. 산포 효과가 수준과 독립으로 실재한다. 부호는 {d}")
        print(f"     다음: lam 부호를 그 방향으로 열고 step12 PART 2 재적합. 그 다음 LB 1회.")
    else:
        print(f"  => ❌ B. 교란이다. step12 의 산포 기울기는 예측 수준을 다시 그린 것이었다.")
        print(f"     다음: B-1(§8.2) 폐기. 산포 축을 더 파지 않는다.")

    # ======================================================== PART 2
    print(BAR)
    print("PART 2  step14 의 '잔여 편향' 을 실제값 구간별로 쪼갠다")
    print("  기전이 맞다면 편향은 최저 구간(0.10~0.20cap)에 몰리고 고출력에서 사라진다.")
    print("  단위 %cap. 괄호는 그 칸의 채점 행수.\n")

    names = [f"{A_EDGES[i]:.2f}~{A_EDGES[i+1]:.2f}" for i in range(len(A_EDGES) - 1)]
    print("    " + f"{'그룹':<10s}" + "".join(f"{n:>16s}" for n in names) + f"{'전체':>12s}")
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float)
        p = post_all[g].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(p) & (a >= 0.10 * cap)
        row = []
        for i in range(len(A_EDGES) - 1):
            m = ok & (a >= A_EDGES[i] * cap) & (a < A_EDGES[i + 1] * cap)
            row.append((np.mean(p[m] - a[m]) / cap * 100 if m.sum() >= MIN_CELL else np.nan,
                        int(m.sum())))
        tot = np.mean(p[ok] - a[ok]) / cap * 100
        print("    " + f"{g.replace('kpx_',''):<10s}"
              + "".join(f"{v:+10.2f}({n:5d})" if np.isfinite(v) else f"{'-':>16s}" for v, n in row)
              + f"{tot:+12.2f}")

    lo = []
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float); p = post_all[g].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(p) & (a >= 0.10 * cap)
        m1 = ok & (a < A_EDGES[1] * cap)
        m2 = ok & (a >= A_EDGES[-2] * cap)
        if m1.sum() >= MIN_CELL and m2.sum() >= MIN_CELL:
            lo.append((np.mean(p[m1] - a[m1]) / cap * 100, np.mean(p[m2] - a[m2]) / cap * 100))
    if lo:
        b_lo = np.mean([v[0] for v in lo]); b_hi = np.mean([v[1] for v in lo])
        print(f"\n  최저 구간 평균 편향 {b_lo:+.2f}%cap   최고 구간 {b_hi:+.2f}%cap")
        if b_lo > 2.0 and b_hi < b_lo / 2:
            print("  => ✅ 기전 확인. step14 의 '잔여 편향' 은 0.17cap floor 가 남긴 흔적이다.")
            print("     §1.3 에 따라 그 floor 는 점수 최적이므로 걷어낼 대상이 아니다.")
            print("     조치: 진단 지표를 '평균 부호편향' 에서 '밴드 적중률' 로 교체한다.")
        else:
            print("  => ⚠️ 기전 불일치. 편향이 저출력에 몰려 있지 않다. step14 를 다시 봐야 한다.")

    print(BAR); print(f"⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()