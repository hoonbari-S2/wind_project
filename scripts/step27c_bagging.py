"""
scripts/step27c_bagging.py
==============================================================================
step27b 가 우연히 드러낸 것 — **적합 데이터를 줄이면 더 좋아짐.** 그리고 후처리 배깅

──────────────────────────────────────────────────────────────────────────────
발견 (사전 등록 아님. **관측**임. §3.2 규칙에 따라 여기서 검정으로 승격시킴)
──────────────────────────────────────────────────────────────────────────────
step27b PART A 월 블록 표의 ②·③ 열을 나란히 보면 이렇게 됨.

    ② 다른 해에서 **절반**만 뽑아 적합 (n ≈ 2,500)
    ③ 다른 해 **전부**로 적합         (n ≈ 10,000)   = 현행 배포 절차
    둘 다 같은 홀드아웃에서 채점함.

      2022·G1  ② 0.6080  ③ 0.6033   ②−③ **+0.0047**
      2022·G2  ② 0.6547  ③ 0.6541          +0.0006
      2023·G1  ② 0.6251  ③ 0.6229          +0.0022
      2023·G2  ② 0.6486  ③ 0.6501          −0.0015
      2023·G3  ② 0.5625  ③ 0.5527   **+0.0098**
      2024·G1  ② 0.6507  ③ 0.6401   **+0.0106**
      2024·G2  ② 0.6599  ③ 0.6577          +0.0022
      2024·G3  ② 0.5935  ③ 0.5935          +0.0000
                                    평균 **+0.0036**, 7/8 비음수

**학습 데이터를 4배 줄였는데 더 나음.** 파라미터 6개짜리 적합에서 이건 정상이 아님.
그리고 이건 연도와 **무관함** — ②·③ 둘 다 '다른 해' 로 적합한 것이고 차이는 오직 표본 수임.
즉 배포 절차(3년 전부로 적합)에 **연도 얘기 없이 바로 적용되는 축**임.

⚠ 2023·G3 는 ②·③ 이 **같은 한 해(2024)** 만 씀 (G3 는 2022 라벨 없음).
   구성이 완전히 같은데 +0.0098 임. 그러므로 '연도 구성 차이' 로는 설명이 안 됨.
   남는 설명은 둘뿐이고, PART 1 이 그 둘을 가름.

──────────────────────────────────────────────────────────────────────────────
PART 1 — 왜 ② > ③ 인가. 설명 후보는 딱 둘임
──────────────────────────────────────────────────────────────────────────────
  (가) **일반화 격차** — ③ 이 적합 목적함수는 제대로 최대화했는데, 그 최적점이
       6%/8% 문턱을 훈련 표본에 맞춰 배치한 것이라 새 표본으로 못 옮겨감.
       n 이 클수록 문턱 배치가 더 정교해지고 더 안 옮겨감. **과적합의 한 형태임.**
  (나) **옵티마이저 실패** — ③ 이 그냥 최적을 못 찾았음. n 이 크면 목적함수 차이가
       평균으로 작아져 `fatol` 에 일찍 걸릴 수 있음.

  가르는 법 — **③ 과 ② 의 곡선을 각각 ③ 의 훈련 집합(전체 풀) 위에서 채점함.**
      obj(③) ≥ obj(②)  ->  ③ 은 제대로 최적화했음. 격차는 (가) 일반화 문제임.
      obj(③) <  obj(②)  ->  (나). ③ 이 최적을 못 찾은 것이므로 **출발점만 늘리면 해결**임.
  추가로 ③ 을 출발점 49개·maxiter 4배로 다시 적합해 obj 가 오르는지 봄.
  안 오르면 (나) 는 확정적으로 배제됨.

  **(가)면 배깅이 답이고, (나)면 출발점이 답임. 처방이 완전히 다르므로 반드시 먼저 가름.**

──────────────────────────────────────────────────────────────────────────────
PART 2 — 후처리 배깅 (사전 등록)
──────────────────────────────────────────────────────────────────────────────
(가) 라면 처방은 정해져 있음. 부분표본으로 K 번 적합해 **곡선을 평균**함.

    out_bag = 단조화( (1/K) Σ_k  적합(부분표본_k) )

각 적합은 서로 다른 행을 봐서 서로 다른 문턱 배치를 하므로, 평균은 표본 특유의
문턱 배치를 지우고 **모든 부분표본이 공유하는 모양만 남김.** 유효 자유도를 줄이되
절점 해상도는 그대로 둠 — §3.15 가 '격자를 성기게' 로 시도했다가 L2·L3 를 깎아
실패한 그 목표를, 해상도를 안 깎고 달성하는 수단임.

§3.17(시드 배깅 폐기)과 다른 축임. 저건 **모델** 시드였고 산포가 1%cap 뿐이라 죽었음.
이건 **후처리 적합**의 표본 배깅이고, 산포는 ②−③ 격차가 이미 +0.0036 로 보여 줬음.

아암 (사전 등록. 전수 탐색 아님)
    cur       f=1.00  K=1              현행
    sub50     f=0.50  K=32   비복원 절반
    sub25     f=0.25  K=32   비복원 1/4   ← ②−③ 관측의 크기와 가장 가까움
    boot      f=1.00  K=32   복원추출(고전적 배깅)
    sub50_k8  f=0.50  K=8    K 의 효과 분리용

판정 (§3.6 규칙 6) 부호 3/3 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수.
평가는 LOYO 중첩, **무가중 대회 점수** (§3.6 규칙 5 범위). 최종 판정은 LB.

  ⚠ 여기서 ✅ 가 떠도 그건 '중첩 추정' 임. §3.6 규칙 4 에 따라 후처리 변화의
     참값은 LB 로만 알 수 있음. 다만 이 축은 **raw 를 안 건드리므로** raw OOF 게이트가
     아예 적용 불가한 종류임(§3.21 부록의 정산구간형 문제와 같음).

──────────────────────────────────────────────────────────────────────────────
연도 축은 어떻게 됐나 — step27b 결론 요약
──────────────────────────────────────────────────────────────────────────────
월 블록 Δ정규 **+0.0051** (누설 제거 후). 무작위 분할의 +0.0143 중 **+0.0092 는 누설**이었음.
그런데 분해가 1−NMAE **+0.0009** / FICR **+0.0092** = FICR 몫 **91%** 임.
사전 등록해 둔 해석 규칙("FICR 몫 90% 초과 + 1−NMAE ≈ 0 이면 문턱 배치")에 걸림.
게다가 8칸 중 5칸만 양수이고 G2 는 −0.0010 으로 사실상 0 임.
=> **연도 가중 축은 종결.** +0.0051 은 '같은 해 데이터를 봤을 때' 의 상한이고,
   2025 데이터는 0행이므로 그중 회수 가능한 몫은 그보다 훨씬 작음.
   step27 [W] 의 +0.0026 은 step27 PART 4 위약 95분위 +0.0013 의 2배지만
   회수 가능 상한 안에서 보면 근거가 못 됨. **PART B(전진연쇄)는 돌리지 않음.**

실행
    # PART 1 만 — 진단. 여기서 (나) 로 나오면 PART 2 는 무의미함
    python scripts/step27c_bagging.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv --only-diag
    # 전체
    python scripts/step27c_bagging.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv
    # 통과 시 제출
    python scripts/step27c_bagging.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv \
        --make-submission sub25 --raw-test ./saved_models/v13/raw_test_preds.csv
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
S9 = ([-0.04, 0.0, 0.04], [0.0, 0.08, 0.15])
S49 = ([-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12],
       [0.0, 0.03, 0.06, 0.09, 0.12, 0.17, 0.22])

# (부분표본 비율, 반복 수, 복원추출 여부)
ARMS = {
    "cur":      (1.00, 1,  False),
    "sub50":    (0.50, 32, False),
    "sub25":    (0.25, 32, False),
    "boot":     (1.00, 16, True),
    "sub50_k8": (0.50, 8,  False),
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


def make_obj(pn, an):
    """적합·진단 양쪽에서 쓰는 목적함수(무가중 대회 점수, 정규화 단위)."""
    i = np.clip(np.searchsorted(KNOTS, pn, side="right") - 1, 0, NK - 2)
    i1 = i + 1
    f = (pn - KNOTS[i]) / (KNOTS[i1] - KNOTS[i])
    om = 1.0 - f
    n = len(an)
    a4 = float(an.sum()) * 4.0

    def obj(x):
        out = mono(x)
        y = np.maximum(out[i] * om + out[i1] * f, 0.0)
        e = np.abs(an - y)
        nmae = float(e.sum() / n)
        price = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
        ficr = float((an * price).sum() / a4) if a4 > 1e-12 else 0.0
        return 0.5 * (1.0 - nmae) + 0.5 * ficr
    return obj


def fit_curve(pn, an, starts=S9, maxiter=500):
    obj = make_obj(pn, an)
    neg = lambda x: -obj(x)
    best_x, best_v = None, -np.inf
    for s, z in itertools.product(*starts):
        x0 = KNOTS + s; x0[0] = z
        r = minimize(neg, mono(x0), method="Nelder-Mead",
                     options=dict(maxiter=maxiter, maxfev=maxiter * 2, xatol=2e-4, fatol=1e-7))
        if np.isfinite(r.fun) and -float(r.fun) > best_v:
            best_v, best_x = -float(r.fun), mono(r.x)
    return best_x


def fit_bagged(pn, an, frac, K, boot, rng, maxiter=500):
    """부분표본 K 개로 적합해 **곡선을 평균**함. K=1·frac=1 이면 현행과 동일."""
    n = len(pn)
    acc = []
    for _ in range(K):
        if K == 1 and frac >= 1.0 and not boot:
            idx = np.arange(n)
        elif boot:
            idx = rng.integers(0, n, size=n)
        else:
            idx = rng.choice(n, size=max(int(n * frac), 200), replace=False)
        c = fit_curve(pn[idx], an[idx], S9, maxiter)
        if c is not None:
            acc.append(c)
    return mono(np.mean(acc, axis=0)) if acc else None


def score_rows(an, pn, cap):
    r = group_score(an * cap, np.clip(pn * cap, 0, cap), cap)
    return (float(r[0]), float(r[1]), float(r[2]))


def judge(sA, sB, targets, years, verbose=True):
    def ytot(s):
        return {y: np.mean([s[(y, g)] for g in targets if (y, g) in s]) for y in years
                if any((y, g) in s for g in targets)}
    ta, tb = ytot(sA), ytot(sB)
    ys = [y for y in years if y in ta and y in tb]
    d = np.array([ta[y] - tb[y] for y in ys])
    mean = float(d.mean()); std = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    same = bool(np.all(d > 0) or np.all(d < 0))
    gpos, gl = 0, []
    for g in targets:
        v = [sA[(y, g)] - sB[(y, g)] for y in years if (y, g) in sA and (y, g) in sB]
        if not v:
            continue
        gpos += int(np.mean(v) > 0); gl.append(f"{gk(g)} {np.mean(v):+.4f}")
    ok = bool(same and mean > 0 and abs(mean) > std and gpos >= 2)
    if verbose:
        print("    연도별 Δ  " + "  ".join(f"{y} {v:+.4f}" for y, v in zip(ys, d)))
        print(f"    평균 {mean:+.4f}  표준편차 {std:.4f}  신뢰비율 "
              f"{(abs(mean)/std if std > 1e-12 else np.inf):5.2f}   그룹별: " + "  ".join(gl))
        print(f"    => {'✅ 채택 후보' if ok else '❌ 채택 안 함'}"
              f"  (부호일치 {same} · 양수 {mean > 0} · |평균|>표준편차 {abs(mean) > std}"
              f" · 양수그룹 {gpos}/{len(targets)})")
    return dict(ok=ok, mean=mean, std=std, gpos=gpos)


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oof", default="./saved_models/v13/oof_preds.csv")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--diag-reps", type=int, default=16)
    ap.add_argument("--only-diag", action="store_true")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--make-submission", default=None, help="아암 이름")
    ap.add_argument("--raw-test", default="./saved_models/v13/raw_test_preds.csv")
    ap.add_argument("--out-name", default=None)
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

    panel = {}
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float) / cap
        p = oof[g].to_numpy(float) / cap
        for y in sorted(set(int(v) for v in df["_year"])):
            m = (df["_year"].to_numpy() == y) & np.isfinite(a) & np.isfinite(p) & (a >= 0.10)
            if m.sum() >= 200:
                panel[(y, g)] = (p[m], a[m])
    years = sorted({y for (y, _) in panel})
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print(BAR)
    print("step27c — '적합 데이터를 줄이면 더 좋아진다' 진단과 후처리 배깅")
    print(f"  OOF {args.oof}   연도 {years}")

    def pool(y, g):
        src = [panel[(t, g)] for t in years if t != y and (t, g) in panel]
        if not src:
            return None
        return (np.concatenate([s[0] for s in src]), np.concatenate([s[1] for s in src]))

    # ================================================================ PART 1
    print(BAR)
    print("PART 1 — ② > ③ 의 정체. (가) 일반화 격차인가 (나) 옵티마이저 실패인가")
    print("  두 곡선을 **③ 의 훈련 집합(전체 풀)** 위에서 채점함. 여기서 갈림.")
    print("    obj(③) ≥ obj(②)  ->  (가) 일반화. 처방 = **배깅**")
    print("    obj(③) <  obj(②)  ->  (나) 옵티마이저. 처방 = **출발점 증설**\n")
    print(f"  {'연도·그룹':<11}{'obj(③)':>10}{'obj(②)평균':>12}{'obj차':>10}"
          f"{'obj(③,49출발)':>14}{'개선':>9}{'평가Δ ②−③':>12}")
    gen, opt, tests = 0, 0, []
    for y in years:
        for g in targets:
            P = pool(y, g)
            if P is None or (y, g) not in panel:
                continue
            ppn, pan = P
            cap = CAPACITY_KWH[g]
            objf = make_obj(ppn, pan)
            c3 = fit_curve(ppn, pan, S9, args.maxiter)
            c3hi = fit_curve(ppn, pan, S49, args.maxiter * 4)
            epn, ean = panel[(y, g)]
            o2, t2 = [], []
            k = max(len(ppn) // 4, 200)
            for _ in range(args.diag_reps):
                jj = rng.choice(len(ppn), size=k, replace=False)
                c2 = fit_curve(ppn[jj], pan[jj], S9, args.maxiter)
                o2.append(objf(c2)); t2.append(score_rows(ean, curve(epn, c2), cap)[0])
            t3 = score_rows(ean, curve(epn, c3), cap)[0]
            d_obj = objf(c3) - float(np.mean(o2))
            d_test = float(np.mean(t2)) - t3
            gen += int(d_obj >= 0); opt += int(d_obj < 0)
            tests.append(d_test)
            print(f"  {str(y) + '·' + gk(g):<11}{objf(c3):10.4f}{np.mean(o2):12.4f}"
                  f"{d_obj:+10.4f}{objf(c3hi):14.4f}{objf(c3hi)-objf(c3):+9.4f}{d_test:+12.4f}")
    tests = np.array(tests)
    print(f"\n  obj(③) ≥ obj(②) 인 칸 {gen}/{gen+opt}   ·   평가Δ(②−③) 평균 {tests.mean():+.4f}"
          f"  양수 {int((tests > 0).sum())}/{len(tests)}")
    if tests.mean() <= 0 or (tests > 0).sum() < len(tests) * 0.5:
        print("  => **(다) 격차 자체가 재현되지 않음.** step27b 의 ②−③ +0.0036 은 여기서 안 나옴.")
        print("     step27b 는 ③ 을 (y,g) 당 **한 번만** 적합한 값과 비교했으므로, 그 한 번이")
        print("     운 나쁜 국소최적이었을 가능성이 큼. 이 진단은 ② 를 여러 번 뽑아 평균하므로")
        print("     그 비대칭이 없음. **관측을 폐기하고 PART 2 도 근거를 잃음.**")
        print("     (그래도 PART 2 는 돌려 봄 — 배깅 자체는 독립적으로 검정 가능한 사전 등록 아암임)")
    elif gen >= opt:
        print("  => **(가) 일반화 격차 확정.** ③ 은 훈련 목적함수를 더 잘 맞췄는데 평가에서 짐.")
        print("     전형적 과적합이고, 이 목적함수에서는 6%/8% 문턱을 훈련 표본에 맞춘 것임.")
        print("     처방은 **배깅** -> PART 2.")
    else:
        print("  => **(나) 옵티마이저 실패.** ③ 이 최적을 못 찾고 있었음.")
        print("     '49출발' 열의 개선폭을 볼 것. 크면 배포 후처리의 출발점부터 늘려야 함.")
        print("     이 경우 PART 2 의 배깅 이득은 대부분 옵티마이저 평균화 효과로 해석해야 함.")
    print(f"  참고: 출발점 49·maxiter 4배로 올렸을 때 obj 개선이 전부 0 에 가까우면")
    print(f"        (나) 는 확정적으로 배제됨 (step27 [S] 의 '9개로 충분' 과 일치).")
    if args.only_diag:
        print(BAR); print(f"경과 {time.time()-t0:.0f}s  (--only-diag)"); return

    # ================================================================ PART 2
    print(BAR)
    print("PART 2 — 후처리 배깅 (사전 등록 아암 5개). LOYO 중첩·무가중 대회 점수")
    print("  ⚠ `boot` 는 매 적합이 전체 크기라 가장 느림(다른 아암의 4배). 전체 5~15분.")
    S = {}
    for name, (frac, K, boot) in ARMS.items():
        sc = {}
        for y in years:
            for g in targets:
                P = pool(y, g)
                if P is None or (y, g) not in panel:
                    continue
                c = fit_bagged(P[0], P[1], frac, K, boot, rng, args.maxiter)
                epn, ean = panel[(y, g)]
                sc[(y, g)] = score_rows(ean, curve(epn, c), CAPACITY_KWH[g])[0]
        S[name] = sc
        tot = np.mean([sc[k] for k in sc])
        print(f"  {name:<10} f={frac:.2f} K={K:<3} {'복원' if boot else '비복원'}"
              f"   중첩 총점 {tot:.4f}   ({time.time()-t0:.0f}s)", flush=True)
    print()
    res = {}
    for name in ARMS:
        if name == "cur":
            continue
        print(f"  [{name}] vs 현행")
        res[name] = judge(S[name], S["cur"], targets, years)
    passed = [k for k, v in res.items() if v["ok"]]
    print(f"\n  통과: {passed if passed else '없음'}")
    if passed:
        win = max(passed, key=lambda k: res[k]["mean"])
        print(f"  **채택 후보: {win}**  (Δ {res[win]['mean']:+.4f} ± {res[win]['std']:.4f})")
        print(f"  ⚠ §3.6 규칙 4 — 후처리 변화의 참값은 LB 로만 알 수 있음. 중첩값은 참고임.")
        print(f"     또한 이 축은 raw 를 안 건드리므로 raw OOF 게이트가 적용 불가함.")
        print(f"\n  제출: python scripts/step27c_bagging.py --config {args.config} "
              f"--oof {args.oof} \\\n          --make-submission {win}")
    else:
        print("  후처리 배깅 축 종결. §3.17(시드 배깅)에 이어 배깅 계열 2번째 폐기로 기록.")

    print(BAR); print(f"경과 {time.time()-t0:.0f}s")

    # ---------------------------------------------------------------- 제출
    if args.make_submission:
        name = args.make_submission
        frac, K, boot = ARMS[name]
        print(BAR); print(f"제출 파일 생성 — {name} (f={frac} K={K} {'복원' if boot else '비복원'})")
        rt = Path(args.raw_test)
        if not rt.exists():
            print(f"  ⛔ {rt} 없음."); return
        sub = pd.read_csv(rt)
        for g in targets:
            src = [panel[(y, g)] for y in years if (y, g) in panel]     # 배포는 전 연도 사용
            pn = np.concatenate([s[0] for s in src]); an = np.concatenate([s[1] for s in src])
            c = fit_bagged(pn, an, frac, K, boot, rng, args.maxiter)
            cap = CAPACITY_KWH[g]
            print(f"  {gk(g)} out_knots = [{', '.join(f'{v:.3f}' for v in c)}]")
            sub[g] = np.clip(curve(sub[g].to_numpy(float) / cap, c) * cap, 0, cap)
        sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        outdir = Path(cfg["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / (args.out_name or f"submit_bag_{name}.csv")
        cols = [c for c in ["forecast_id", "forecast_kst_dtm"] if c in sub.columns] + targets
        sub[cols].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  🚀 {out}   행 {len(sub):,}  결측 {int(sub[targets].isna().sum().sum())}")
        for g in targets:
            v = sub[g].to_numpy(float)
            print(f"     {g}: 평균 {np.mean(v)/CAPACITY_KWH[g]*100:5.1f}%cap  "
                  f"최대 {np.max(v)/CAPACITY_KWH[g]*100:5.1f}%cap")


if __name__ == "__main__":
    main()