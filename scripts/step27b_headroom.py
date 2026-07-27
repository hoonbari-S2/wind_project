"""
scripts/step27b_headroom.py
==============================================================================
step27 PART 0 의 +0.0203 을 **분해**함 — 그게 진짜 여지인가 인-샘플 착시인가

──────────────────────────────────────────────────────────────────────────────
왜 이 스크립트가 필요한가
──────────────────────────────────────────────────────────────────────────────
step27 PART 0 의 '오라클' 은 **평가 연도로 직접 적합해 그 연도에서 점수를 잰 값**임.
즉 인-샘플임. 그래서 그 +0.0203 은 두 개가 섞인 값임.

    (1) 진짜 연도 불일치  — 최적 곡선이 해마다 달라서 생기는 손실. **되찾을 수 있음.**
    (2) 인-샘플 착시      — 파라미터 6개를 평가 연도에 맞춘 낙관 편향. **되찾을 수 없음.**

둘을 안 가르면 PART 2 의 [L] +0.0041 · [W] +0.0026 이 어느 크기의 못에 담긴 물인지 모름.
그리고 이건 논증으로 가를 문제가 아니라 **재면 되는 문제**임.

의심스러운 정황 — 연도별 여지가 +0.0203 / +0.0198 / +0.0207 로 **거의 상수**임.
진짜 연도 불일치라면 해마다 달라야 함. 상수는 (2) 의 냄새임.
반대 정황 — G1 +0.0239 vs G2 +0.0091 인데 두 그룹의 채점행 수가 거의 같음.
착시라면 n 이 같을 때 같아야 하므로 **G1 에는 (1) 이 있다는 뜻**임.
=> 정황이 양쪽으로 갈림. 그래서 잼.

──────────────────────────────────────────────────────────────────────────────
PART A — 편향 없는 여지 측정 (이 스크립트의 핵심)
──────────────────────────────────────────────────────────────────────────────
연도 Y 의 채점행을 무작위 반으로 쪼갬(A / B). **세 곡선을 같은 B 에서 채점함.**

    ① 같은해½      Y 의 절반 A 로 적합                         n = n/2
    ② 다른해同量    다른 연도에서 **같은 행 수 n/2** 만 뽑아 적합   n = n/2
    ③ 중첩(2년)     Y 를 뺀 나머지 전부로 적합                    n = 2년  = 현행 배포 절차

    **Δ정규 = ①−②**   ← 핵심 숫자
       Δ실무 = ①−③

**셋 다 B 에 대해 out-of-sample 이므로 인-샘플 착시가 정의상 0 임.**

② 가 왜 필요한가 — **첫 판에는 ① vs ③ 만 넣었는데 그건 교란된 비교였음.**
①−③ 이 음수로 나와도 그게 '연도끼리 잘 전이된다' 인지 '① 이 그냥 데이터가 4배
적어서 진 것' 인지 구분이 안 됨. ② 는 **① 과 학습 표본 크기가 같고 연도만 다름.**
그래서 ①−② 는 크기 교란이 제거된, 오직 '어느 해 데이터인가' 만의 효과임.

    Δ정규 ≥ +0.005   연도 불일치 실재 -> PART B·C 진행
    Δ정규 ≈  0       같은 해든 다른 해든 같음. PART 0 의 +0.0203 은 통째로 착시. **축 종결**
    Δ정규 <  0       크기가 같은데 다른 해가 나음. 분할 표준편차와 대조해 잡음인지 볼 것

**추가 분해 — Δ정규 를 1−NMAE 와 FICR 로 쪼갬.**
FICR 만 움직이고 1−NMAE 가 0 이면 곡선 모양이 실제로 다른 게 아니라 **6%/8% 문턱
근처 행을 밀어 넣은 것**뿐임. 그건 그 해에만 붙는 성질이라 다른 해로 못 옮겨감.
둘 다 움직여야 진짜 연도 불일치임.

분할을 R 번 반복해 분할 잡음을 평균으로 지움. 방향(A→B, B→A) 둘 다 씀.

──────────────────────────────────────────────────────────────────────────────
PART B — LOYO 는 recency 가설에 **구조적으로 불리한 계측기**임
──────────────────────────────────────────────────────────────────────────────
step27 [W] recency 의 연도별 Δ 를 다시 봄.

    2022 폴드  −0.0003   (2023·2024 로 적합해 2022 예측 — recency = "미래를 더 믿어라")
    2023 폴드  +0.0008   (2022·2024 로 적합해 2023 예측 — 양쪽이 인접, 중립)
    2024 폴드  **+0.0072**   (2022·2023 로 적합해 2024 예측 — **이것만 실제 배치와 같은 모양**)

실제 배치는 2022·2023·2024 로 적합해 **2025** 를 예측함. 즉 항상 '학습창 다음 해' 임.
LOYO 3폴드 중 그 모양인 것은 **2024 폴드 하나뿐**이고, 나머지 둘에서 recency 는
말이 안 되는 지시(2022 를 예측하는데 2024 를 가장 믿으라)를 함.

**그러므로 [W] 에 3/3 부호일치를 요구한 사전 등록은 축에 맞지 않는 게이트였음.**
step16 때 부호 방향을 안 적은 것과 같은 급의 설계 실수임 — 규칙을 어긴 게 아니라
**규칙을 잘못 고른** 것임. 사후 구제가 아니라 계측기를 바꿔야 하는 사안이므로,
바꾼 계측기로 다시 재고 그 결과에 따름.

전진연쇄(rolling origin) — **모든 폴드가 배치와 같은 모양**임.
    적합 {2022}            → 평가 2023
    적합 {2022, 2023}      → 평가 2024
폴드가 2개뿐이라 검정력이 약함. 그래서 사전 등록 조건을
`2/2 양수 ∧ 평균 > 위약 95분위` 로 두고, **최종 판정은 LB** 로 넘김 (§3.6 규칙 4·5).

──────────────────────────────────────────────────────────────────────────────
PART C — 그룹 선택적 축소. §3.22 에서 나온 사전 예측
──────────────────────────────────────────────────────────────────────────────
step27 [L] λ=0.5 는 게이트를 놓쳤지만(2022 −0.0015) **그룹 3/3 양수**였고
G3 가 최대(+0.0070)였음. §3.15 의 `[관측, 쫓지 않음]`(축소 격자에서 G3 만 양수)과
방향이 정확히 같음. 사전 예측이 맞은 것이므로 버리기 아까움.

그런데 §3.22 는 **G1↔G2 는 동질, G3 는 짝이 없음** 을 확정했음
(G1 이 G2 와 묶일 때 +0.0126, G3 와 묶일 때 +0.0034).
전역 λ 는 그 구조를 무시하고 G1 을 G3 쪽으로도 끌어당김. 실제로 유일한 음수 폴드인
2022 는 **G3 라벨이 없어 평가가 G1·G2 만으로 이뤄진 폴드**임 — 동질 쌍만 볼 때
전역 축소가 손해였다는 뜻으로 읽힘.

사전 등록 예측: **`g1g2` (G1·G2 끼리만 묶고 G3 는 독립) 가 전역 λ=0.5 를 이긴다.**
   틀리면 §3.22 의 동질성 구조가 후처리 층에는 전이되지 않는다는 뜻이고 그것도 결론임.

아암 (λ 벡터 · 결합 대상)
    cur      (1, 1, 1)                        현행
    all50    (.5, .5, .5) → pooled(G1,G2,G3)  step27 [L]
    g3only   (1, 1, .5)  → pooled(G1,G2,G3)   G3 만 빌림
    g1g2     (.5, .5, 1) → pooled(G1,G2)      **동질 쌍끼리만.** ← 사전 예측 1순위
    g1g2_g3  (.5,.5,.5): G1·G2 는 pooled(G1,G2), G3 는 pooled(전체)   혼합형

──────────────────────────────────────────────────────────────────────────────
PART D — λ 축 전용 위약
step27 의 위약은 **연도 가중 축** 것이라 λ 축에 그대로 못 씀.
무작위 λ 벡터(그룹별 독립 U[0,1])로 같은 게이트를 돌려 λ 축의 우연 통과율을 따로 잼.
──────────────────────────────────────────────────────────────────────────────

실행 (학습 0회)
    python scripts/step27b_headroom.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv
    # PART A 만 (2~3분). 여기서 Δ ≈ 0 이면 나머지는 볼 필요 없음
    python scripts/step27b_headroom.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv --only-a
==============================================================================
"""
import argparse
import itertools
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
import src.postprocessing as PP

quiet_warnings()
BAR = "=" * 92

KNOTS = np.asarray(getattr(PP, "KNOTS", [0.0, 0.15, 0.35, 0.60, 0.85, 1.00]), dtype=float)
NK = len(KNOTS)
STARTS = ([-0.04, 0.0, 0.04], [0.0, 0.08, 0.15])          # step27 이 9개로 충분함을 확인함
WEIGHTS = {
    "uniform":        {2022: 1.0, 2023: 1.0, 2024: 1.0},
    "recency":        {2022: 1.0, 2023: 2.0, 2024: 3.0},
    "recency_strong": {2022: 1.0, 2023: 2.0, 2024: 4.0},
}
# PART C 아암 — (그룹별 λ, 그룹별 pooled 대상 그룹들)
ARMS = {
    "cur":     (dict(G1=1.0, G2=1.0, G3=1.0), None),
    "all50":   (dict(G1=0.5, G2=0.5, G3=0.5), {"G1": "ALL", "G2": "ALL", "G3": "ALL"}),
    "g3only":  (dict(G1=1.0, G2=1.0, G3=0.5), {"G3": "ALL"}),
    "g1g2":    (dict(G1=0.5, G2=0.5, G3=1.0), {"G1": "G1G2", "G2": "G1G2"}),
    "g1g2_g3": (dict(G1=0.5, G2=0.5, G3=0.5), {"G1": "G1G2", "G2": "G1G2", "G3": "ALL"}),
}


def gk(g):
    return "G" + g.split("_")[-1]


def mono(x):
    return np.maximum.accumulate(np.clip(np.asarray(x, float), 0.0, 1.4))


def curve(pn, out):
    y = np.interp(pn, KNOTS, out)
    hi = pn > KNOTS[-1]
    if np.any(hi):
        sl = (out[-1] - out[-2]) / (KNOTS[-1] - KNOTS[-2])
        y = np.where(hi, out[-1] + sl * (pn - KNOTS[-1]), y)
    return np.clip(y, 0.0, None)


def fit_curve(pn, an, w, maxiter=500):
    i = np.clip(np.searchsorted(KNOTS, pn, side="right") - 1, 0, NK - 2)
    i1 = i + 1
    f = (pn - KNOTS[i]) / (KNOTS[i1] - KNOTS[i])
    om = 1.0 - f
    sw = float(w.sum()); wa = w * an; wa4 = float(wa.sum()) * 4.0

    def neg(x):
        out = mono(x)
        y = np.maximum(out[i] * om + out[i1] * f, 0.0)
        e = np.abs(an - y)
        nmae = float((w * e).sum() / sw)
        price = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
        ficr = float((wa * price).sum() / wa4) if wa4 > 1e-12 else 0.0
        return -(0.5 * (1.0 - nmae) + 0.5 * ficr)

    best_x, best_v = None, -np.inf
    for s, z in itertools.product(*STARTS):
        x0 = KNOTS + s; x0[0] = z
        r = minimize(neg, mono(x0), method="Nelder-Mead",
                     options=dict(maxiter=maxiter, maxfev=maxiter * 2, xatol=2e-4, fatol=1e-7))
        if np.isfinite(r.fun) and -float(r.fun) > best_v:
            best_v, best_x = -float(r.fun), mono(r.x)
    return best_x


def score_rows(an, pn, cap):
    """정규화 단위 배열 -> 프로젝트 표준 group_score (총점, 1−NMAE, FICR). 무가중."""
    r = group_score(an * cap, np.clip(pn * cap, 0, cap), cap)
    return (float(r[0]), float(r[1]), float(r[2]))


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oof", default="./saved_models/v13/oof_preds.csv")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--splits", type=int, default=8, help="PART A 무작위 분할 반복")
    ap.add_argument("--placebo", type=int, default=25)
    ap.add_argument("--only-a", action="store_true")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    df["_year"] = pd.to_datetime(df["forecast_kst_dtm"]).dt.year
    answer = df[targets].copy()
    oof = pd.read_csv(args.oof, index_col=0).reindex(index=df.index, columns=targets).astype(float)

    # 시간 블록 분할용 — 연중 일수. 무작위 분할의 자기상관 누설을 막는 데 씀.
    doy = pd.to_datetime(df["forecast_kst_dtm"]).dt.dayofyear.to_numpy()

    # 채점행만, 정규화 단위로 패널 구성
    panel = {}
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float) / cap
        p = oof[g].to_numpy(float) / cap
        for y in sorted(set(int(v) for v in df["_year"])):
            m = (df["_year"].to_numpy() == y) & np.isfinite(a) & np.isfinite(p) & (a >= 0.10)
            if m.sum() >= 200:
                panel[(y, g)] = (p[m], a[m], doy[m])
    years = sorted({y for (y, _) in panel})

    print(BAR)
    print("step27b — PART 0 의 +0.0203 분해")
    print(f"  OOF {args.oof}   연도 {years}   절점 {NK}개/그룹")
    print("  채점행  " + "  ".join(f"{y}·{gk(g)} {len(panel[(y,g)][0]):,}"
                                   for y in years for g in targets if (y, g) in panel))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---------------------------------------------------------------- 중첩 곡선 (공통)
    nested = {}     # (fold_year, group) -> out_knots  (그 해를 뺀 나머지로 적합)
    for y in years:
        for g in targets:
            src = [panel[(t, g)] for t in years if t != y and (t, g) in panel]
            if not src:
                continue
            pn = np.concatenate([s[0] for s in src]); an = np.concatenate([s[1] for s in src])
            nested[(y, g)] = fit_curve(pn, an, np.ones(len(pn)), args.maxiter)

    # ================================================================ PART A
    print(BAR)
    print("PART A — 편향 없는 여지. **크기를 맞춘** 같은해 vs 다른해, 같은 홀드아웃에서 채점")
    print("  세 곡선을 **같은 홀드아웃 B** 에서 채점함. 전부 B 에 대해 out-of-sample 이라 착시 = 0.")
    print("    ① 같은해½    Y 의 나머지 절반으로 적합            n = n/2")
    print("    ② 다른해同量  다른 연도에서 **같은 행 수**만 뽑아 적합  n = n/2   ← 크기 맞춤 대조군")
    print("    ③ 중첩(2년)   Y 를 뺀 나머지 전부로 적합            n = 2년   = 현행 배포 절차")
    print("\n  **Δ정규 = ① − ②** 가 이 스크립트의 핵심 숫자임.")
    print("    두 곡선의 학습 표본 크기가 같으므로 차이는 오직 '**어느 해 데이터인가**' 뿐임.\n")
    print("  ⚠ **분할 방식이 결과를 좌우함 — 무작위 행 분할에는 자기상관 누설이 있음.**")
    print("    풍속은 시간 자기상관이 매우 강해서 t 시와 t+1 시는 거의 같은 상태임.")
    print("    무작위로 행을 반씩 나누면 A 안에 B 행의 **사실상 복제본**이 들어감.")
    print("    절점 6개짜리 전역 사상이라도 6%/8% 문턱 배치는 그 복제본을 타고 전이됨.")
    print("    ② 는 다른 해에서 뽑으므로 그런 복제본이 없음 -> **①만 유리한 비대칭**임.")
    print("    그래서 시간 블록 분할(블록 단위로 A/B 배정)을 같이 돌려 비교함.")
    print("    무작위 − 블록 = **자기상관 누설의 크기 추정치**임.\n")

    def run_mode(name, block_days):
        A, A2, comp, rows = {}, {}, [], []
        for y in years:
            for g in targets:
                if (y, g) not in panel or (y, g) not in nested:
                    continue
                pn, an, dy = panel[(y, g)]
                cap = CAPACITY_KWH[g]
                n = len(pn)
                src = [panel[(t, g)] for t in years if t != y and (t, g) in panel]
                opn = np.concatenate([s[0] for s in src]); oan = np.concatenate([s[1] for s in src])
                blk = None if block_days is None else (dy // block_days).astype(int)
                ub = None if blk is None else np.unique(blk)
                d1, d2, s1l, s2l, s3l, cm = [], [], [], [], [], []
                for r in range(args.splits):
                    if blk is None:                       # 무작위 행 분할
                        perm = rng.permutation(n); h = n // 2
                        pairs = [(perm[:h], perm[h:]), (perm[h:], perm[:h])]
                    else:                                 # 블록 단위 배정
                        pb = rng.permutation(len(ub)); h = len(ub) // 2
                        mA = np.isin(blk, ub[pb[:h]])
                        iA, iB = np.where(mA)[0], np.where(~mA)[0]
                        pairs = [(iA, iB), (iB, iA)]
                    for lo, hi in pairs:
                        if len(lo) < 200 or len(hi) < 200:
                            continue
                        k = min(len(lo), len(opn))
                        jj = rng.choice(len(opn), size=k, replace=False)
                        c1 = fit_curve(pn[lo], an[lo], np.ones(len(lo)), args.maxiter)
                        c2 = fit_curve(opn[jj], oan[jj], np.ones(k), args.maxiter)
                        r1 = score_rows(an[hi], curve(pn[hi], c1), cap)
                        r2 = score_rows(an[hi], curve(pn[hi], c2), cap)
                        r3 = score_rows(an[hi], curve(pn[hi], nested[(y, g)]), cap)
                        d1.append(r1[0] - r2[0]); d2.append(r1[0] - r3[0])
                        s1l.append(r1[0]); s2l.append(r2[0]); s3l.append(r3[0])
                        cm.append((r1[1] - r2[1], r1[2] - r2[2]))
                if not d1:
                    continue
                d1, d2 = np.array(d1), np.array(d2)
                A[(y, g)] = (float(d1.mean()), float(d1.std(ddof=1)))
                A2[(y, g)] = float(d2.mean())
                comp.append(np.mean(cm, axis=0))
                rows.append((f"{y}·{gk(g)}", np.mean(s1l), np.mean(s2l), np.mean(s3l),
                             d1.mean(), d2.mean(), d1.std(ddof=1)))
        return A, A2, np.mean(comp, axis=0), rows

    MODES = [("무작위 행", None), ("주 블록(7일)", 7), ("월 블록(30일)", 30)]
    RES = {}
    for name, bd in MODES:
        A, A2, cm, rows = run_mode(name, bd)
        RES[name] = (A, A2, cm)
        am = np.array([v[0] for v in A.values()])
        print(f"  [{name}]" + ("   ← 누설 있음. 참고용" if bd is None else "   ← 누설 제거"))
        print(f"  {'연도·그룹':<11}{'①같은해½':>10}{'②다른해同量':>13}{'③중첩2년':>11}"
              f"{'Δ정규①−②':>12}{'①−③':>10}{'표준편차':>10}")
        for nm, a1, a2v, a3, dd1, dd2, sd in rows:
            print(f"  {nm:<11}{a1:10.4f}{a2v:13.4f}{a3:11.4f}{dd1:+12.4f}{dd2:+10.4f}{sd:10.4f}")
        print(f"  {'평균':<11}{'':>10}{'':>13}{'':>11}{am.mean():+12.4f}"
              f"{np.mean(list(A2.values())):+10.4f}"
              f"     양수 {int((am > 0).sum())}/{len(am)}")
        print(f"  분해: 1−NMAE {cm[0]:+.4f}   FICR {cm[1]:+.4f}"
              f"   (FICR 몫 {abs(cm[1])/max(abs(cm[0])+abs(cm[1]),1e-9)*100:.0f}%)\n")

    rnd = np.mean([v[0] for v in RES["무작위 행"][0].values()])
    wk = np.mean([v[0] for v in RES["주 블록(7일)"][0].values()])
    mo = np.mean([v[0] for v in RES["월 블록(30일)"][0].values()])
    print(f"  **누설 추정 — 무작위 {rnd:+.4f}  주블록 {wk:+.4f}  월블록 {mo:+.4f}**")
    print(f"    무작위 − 월블록 = {rnd - mo:+.4f}  <- 자기상관 누설로 부풀려진 몫")
    real = min(wk, mo)
    print(f"    **누설 제거 후 실제 여지 ≈ {real:+.4f}**")
    for g in targets:
        v = [RES["월 블록(30일)"][0][(y, g)][0] for y in years
             if (y, g) in RES["월 블록(30일)"][0]]
        if v:
            print(f"     {gk(g)} 월블록 Δ정규 {np.mean(v):+.4f}")
    print(f"\n  해석 기준 — **월/주 블록 값으로 읽을 것. 무작위는 참고용임**")
    print(f"    ≥ +0.005  연도 불일치 실재 -> PART B·C 진행")
    print(f"    ≈  0.000  연도는 서로 잘 전이됨. step27 PART 0 의 +0.0203 은 착시. **축 종결**")
    print(f"    또한 FICR 몫이 90% 를 넘고 1−NMAE 가 0 근처면, 남은 것도 문턱 배치라")
    print(f"    다른 해로 옮겨가지 않음. **블록 분할에서도 1−NMAE 가 양수여야 진짜임.**")
    if real < 0.005:
        print(f"\n  🛑 누설 제거 후 여지가 +0.005 미만임. [L]·[W] 의 +0.004 급 효과는")
        print(f"     담길 못이 없음. PART B·C 를 돌리더라도 제출 근거로는 못 씀.")
    if args.only_a:
        print(BAR); print(f"경과 {time.time()-t0:.0f}s  (--only-a 라 여기서 멈춤)"); return

    # ================================================================ PART B
    print(BAR)
    print("PART B — 전진연쇄. **모든 폴드가 실제 배치(학습창 다음 해)와 같은 모양**")
    print("  LOYO 는 3폴드 중 2폴드에서 recency 에 말이 안 되는 지시를 함. 계측기를 바꿈.")
    chains = [([years[i] for i in range(k + 1)], years[k + 1]) for k in range(len(years) - 1)]
    print("  폴드: " + "   ".join(f"적합{f} → 평가 {e}" for f, e in chains))
    fwd = {}
    for wn, wmap in WEIGHTS.items():
        rows = {}
        for fit_years, ev in chains:
            for g in targets:
                src = [(panel[(t, g)], wmap.get(t, 1.0)) for t in fit_years if (t, g) in panel]
                if not src or (ev, g) not in panel:
                    continue
                pn = np.concatenate([s[0][0] for s in src])
                an = np.concatenate([s[0][1] for s in src])
                w = np.concatenate([np.full(len(s[0][0]), s[1]) for s in src])
                c = fit_curve(pn, an, w, args.maxiter)
                epn, ean = panel[(ev, g)][0], panel[(ev, g)][1]
                rows[(ev, g)] = score_rows(ean, curve(epn, c), CAPACITY_KWH[g])[0]
        fwd[wn] = rows
    base = fwd["uniform"]
    print(f"\n  {'가중':<16}" + "".join(f"{'평가 ' + str(e):>14}" for _, e in chains) + f"{'평균 Δ':>12}")
    for wn in WEIGHTS:
        cells, dall = [], []
        for _, e in chains:
            d = [fwd[wn][(e, g)] - base[(e, g)] for g in targets if (e, g) in base]
            cells.append(np.mean(d) if d else np.nan); dall += d
        mark = "  ← 기준" if wn == "uniform" else ""
        print(f"  {wn:<16}" + "".join(f"{c:+14.4f}" for c in cells)
              + f"{np.nanmean(cells):+12.4f}" + mark)
    for wn in ["recency", "recency_strong"]:
        cells = [np.mean([fwd[wn][(e, g)] - base[(e, g)] for g in targets if (e, g) in base])
                 for _, e in chains]
        gl = "  ".join(f"{gk(g)} "
                       f"{np.mean([fwd[wn][(e,g)]-base[(e,g)] for _, e in chains if (e,g) in base]):+.4f}"
                       for g in targets)
        print(f"\n  [{wn}] 폴드별 {['%+.4f' % c for c in cells]}  그룹별 {gl}")
        print(f"     사전 등록: 2/2 양수 ∧ 평균 > 위약 95분위  =>  "
              f"{'✅ 조건 1 통과' if all(c > 0 for c in cells) else '❌ 폴드 부호 갈림'}")

    # ================================================================ PART C
    print(BAR)
    print("PART C — 그룹 선택적 축소. **사전 예측: `g1g2` 가 `all50` 을 이긴다** (§3.22 동질성)")
    pooled = {}
    for y in years:
        for tag, gs in [("ALL", targets), ("G1G2", [g for g in targets if gk(g) in ("G1", "G2")])]:
            src = []
            for g in gs:
                s = [panel[(t, g)] for t in years if t != y and (t, g) in panel]
                if not s:
                    continue
                pn = np.concatenate([x[0] for x in s]); an = np.concatenate([x[1] for x in s])
                src.append((pn, an, np.full(len(pn), 1.0 / len(pn))))   # 그룹별 총가중 균등
            if src:
                pooled[(y, tag)] = fit_curve(np.concatenate([s[0] for s in src]),
                                             np.concatenate([s[1] for s in src]),
                                             np.concatenate([s[2] for s in src]), args.maxiter)

    def arm_scores(lam, tgt):
        out = {}
        for y in years:
            for g in targets:
                if (y, g) not in panel or (y, g) not in nested:
                    continue
                l = lam[gk(g)]
                pf = pooled.get((y, (tgt or {}).get(gk(g), "ALL")))
                c = nested[(y, g)] if (l >= 1.0 or pf is None) else mono(l * nested[(y, g)] + (1 - l) * pf)
                pn, an = panel[(y, g)][0], panel[(y, g)][1]
                out[(y, g)] = score_rows(an, curve(pn, c), CAPACITY_KWH[g])[0]
        return out

    S = {k: arm_scores(*v) for k, v in ARMS.items()}
    cur = S["cur"]

    def ytot(s):
        return {y: np.mean([s[(y, g)] for g in targets if (y, g) in s]) for y in years}

    print(f"\n  {'아암':<10}" + "".join(f"{y:>10}" for y in years)
          + f"{'평균Δ':>10}{'표준편차':>10}{'양수그룹':>10}")
    res = {}
    for k in ARMS:
        ta, tb = ytot(S[k]), ytot(cur)
        d = np.array([ta[y] - tb[y] for y in years])
        gp = sum(int(np.mean([S[k][(y, g)] - cur[(y, g)]
                              for y in years if (y, g) in cur]) > 0) for g in targets)
        res[k] = (d.mean(), d.std(ddof=1), gp, bool(np.all(d > 0)))
        mark = "  ← 현행" if k == "cur" else ("  ← 사전 예측 1순위" if k == "g1g2" else "")
        print(f"  {k:<10}" + "".join(f"{v:+10.4f}" for v in d)
              + f"{d.mean():+10.4f}{d.std(ddof=1):10.4f}{gp:>7}/3" + mark)
    print(f"\n  사전 예측 검정: g1g2 {res['g1g2'][0]:+.4f}  vs  all50 {res['all50'][0]:+.4f}"
          f"   =>  {'✅ 예측 적중' if res['g1g2'][0] > res['all50'][0] else '❌ 예측 빗나감'}")
    if res["g1g2"][0] <= res["all50"][0]:
        print("     §3.22 의 동질성 구조가 후처리 층으로는 전이되지 않음. 그것도 결론이므로 기록할 것.")

    # ================================================================ PART D
    print(BAR)
    print(f"PART D — **λ 축 전용** 위약 {args.placebo}개 (step27 위약은 연도 가중 축 것이라 못 씀)")
    npass, dl = 0, []
    for _ in range(args.placebo):
        lam = {gk(g): float(rng.uniform(0.0, 1.0)) for g in targets}
        s = arm_scores(lam, {gk(g): "ALL" for g in targets})
        ta, tb = ytot(s), ytot(cur)
        d = np.array([ta[y] - tb[y] for y in years])
        gp = sum(int(np.mean([s[(y, g)] - cur[(y, g)] for y in years if (y, g) in cur]) > 0)
                 for g in targets)
        ok = bool(np.all(d > 0) and d.mean() > 0 and abs(d.mean()) > d.std(ddof=1) and gp >= 2)
        npass += int(ok); dl.append(d.mean())
    dl = np.array(dl)
    q95 = float(np.quantile(dl, 0.95))
    print(f"  우연 통과 {npass}/{args.placebo} = **{npass/args.placebo*100:.0f}%**")
    print(f"  위약 Δ  평균 {dl.mean():+.4f}  표준편차 {dl.std(ddof=1):.4f}  95분위 {q95:+.4f}")
    print(f"  => **λ 축에서 {q95:+.4f} 이하는 근거가 못 됨.**")
    for k in ARMS:
        if k == "cur":
            continue
        print(f"     {k:<10} Δ {res[k][0]:+.4f}  "
              f"{'여전히 유효' if res[k][0] > q95 else '위약 범위 안 — 폐기'}")

    print(BAR)
    print("종합 판단 순서")
    print("  1. PART A 의 Δ 가 못의 크기임. 여기가 0 이면 아래는 전부 의미 없음.")
    print("  2. PART B 는 recency 를 배치와 같은 모양으로 다시 잼. LOYO 결과를 대체함.")
    print("  3. PART C·D 는 λ 를 §3.22 구조에 맞춰 다시 잼.")
    print("  4. 통과해도 **최종 판정은 LB** 임 (§3.6 규칙 4). 제출은 1회씩 따로.")
    print(f"경과 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()