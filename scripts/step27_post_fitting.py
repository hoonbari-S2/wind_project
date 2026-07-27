"""
scripts/step27_post_fitting.py
==============================================================================
후처리 **적합 절차** 격자탐색 — 연도 가중 · 그룹 간 축소 · 다중 출발점

──────────────────────────────────────────────────────────────────────────────
이 실험이 §3.14 와 모순되지 않는 이유 (먼저 읽을 것)
──────────────────────────────────────────────────────────────────────────────
§3.14 는 **후처리 축이 닫혔다** 고 썼고 그것은 지금도 맞음. 단 닫힌 것은 정확히
**함수 공간** 이었음.

  닫힌 것 (건드리지 않음)
    * 사상의 해상도            step16 / §3.15 — 증설·축소 5개 전부 하락
    * 행별 산포 변조           step12 / §3.14 — 예측 수준 교란
    * 시각·계절 버킷 배율      step14 / §3.16 — m = 1.000
    * 그룹별 강도 **분리**     B-3 / §8.2 — 총점 +0.0004, 노이즈 밑
    => 전부 "같은 p 의 함수를 어떤 모양으로 그릴 것인가" 였음.

  아직 한 번도 안 건드린 것 (이 실험)
    * **추정량** — 같은 함수 공간 안에서 **그 곡선을 어떤 표본으로 어떻게 맞출 것인가.**
      (a) 적합이 보는 행에 연도 가중을 줄 것인가
      (b) 세 그룹의 곡선이 서로 **힘을 빌릴** 것인가 (축소 — B-3 의 정확한 반대 방향)
      (c) 옵티마이저가 애초에 최적을 찾고 있기는 한가 (출발점 개수)

  (b) 가 B-3 의 재탕이 아닌 이유. B-3 은 그룹에 자유도를 **더** 줬고 실패했음.
  §3.15 는 정반대의 관측을 남겼음 — **축소 격자 3개 전부에서 G3 만 양수**
  (+0.0005 / +0.0012 / +0.0005). 라벨이 2년뿐인 G3 는 파라미터가 적은 쪽이 유리함.
  그런데 step16 이 시도한 파라미터 축소 수단은 '격자를 성기게' 였고, 그건 행의 95%가
  있는 L2·L3 의 해상도까지 같이 깎아서 총점을 떨어뜨렸음(§3.15).
  **축소(shrinkage)는 L2·L3 해상도를 유지한 채 유효 자유도만 줄이는 수단임.**
  즉 §3.15 의 `[관측, 쫓지 않음]` 을 깎지 않고 쫓는 유일한 방법임.

  (c) 는 **메타 점검**임. 현행은 출발점 9개(s∈{−0.04,0,0.04} × z∈{0,0.08,0.15})임.
  목적함수는 가격 계단(4.0/3.0/0) 때문에 **불연속·다봉**이라 국소최적이 실재함.
  만약 출발점을 늘려 적합값이 유의하게 움직이면 §3.14~3.16 의 음수 판정들에
  옵티마이저 노이즈가 섞여 있었다는 뜻이므로 그 결론들에 오차막대를 붙여야 함.
  움직이지 않으면 그 결론들이 오히려 **더 단단해짐.** 어느 쪽이든 정보임.

──────────────────────────────────────────────────────────────────────────────
연도 가중을 왜 기상자료로 정당화하는가 — 그리고 어디까지만 정당한가
──────────────────────────────────────────────────────────────────────────────
기상청 강원 연 기후 특성에서 확인된 것(§3.9 가 램프 가설로는 기각했으나 **사실 자체는 살아 있음**)

    2023  연평균기온 12.1℃ — 역대 1위, 평년 대비 +1.3℃
          연강수량 1437.0mm (평년비 104.4%, 상위 18위)
          **월 기온변동폭 11월 6.4℃(1위) · 12월 6.1℃(1위) · 1월 4.5℃(10위)**
          12월 강수량 평년비 449.0% — 역대 1위 / 3월 +3.6℃(1위) / 9월 +2.1℃(1위)
          8/9–10 태풍 카눈
    (§3.9 표에 이미 적힌 겨울 기온변동 평균: 2022 = 3.73 / 2023 = 6.17 / 2024 = 4.07)

우리 쪽 관측과 맞물림 — v13 raw OOF 연도별 **2022 0.6039 / 2023 0.5919 / 2024 0.6121.**
2023 은 어느 학습원을 쓰든 일관되게 가장 어려웠음(§3.1, 연도 표준편차 0.0093 으로 최저).
=> **2023 은 "평범한 해" 가 아님.** 후처리 곡선을 세 해로 균등 적합하면 그 곡선은
   1/3 만큼 이상기후 연도에 맞춰짐.

**여기서 선을 그어야 함 (§6 규칙 3).**
  * ✅ 허용 — "2023 은 이상치였으니 적합에서 비중을 낮춘다".
       2022·2023 년판 PDF 는 전부 2025 년 예측 시점 **이전** 에 발간됐고,
       무엇보다 이건 **피처가 아니라 학습 표본 가중** 임. 테스트 행을 만지지 않음.
  * ✅ 허용 — "최근 연도에 더 큰 가중" (설비·정비 이력 드리프트에 대한 사전분포).
  * ❌ 금지 — "2025 가 어느 해와 닮았으니 그 해에 가중". **2025 년판 PDF 는 존재하지 않음.**
       테스트 연도의 기후를 안다고 가정하는 순간 누설임. 이 스크립트는 그 형태를 만들지 않음.

그리고 이 논증은 **가설이지 근거가 아님.** 그래서 아래에 `y2023_up`(2023 을 오히려 **키우는**)
를 **부호 대조군** 으로 같이 넣었음. down 과 up 이 둘 다 개선으로 나오면 그 축은 신호가
아니라 잡음임. 이건 사후 변명이 아니라 사전 등록임.

──────────────────────────────────────────────────────────────────────────────
격자 (사전 등록)
──────────────────────────────────────────────────────────────────────────────
  축 W  연도 가중   uniform(현행) / recency(1,2,3) / recency_strong(1,2,4)
                    / y2023_down(1,0.5,1) / y2023_up(1,2,1)←부호 대조군
  축 L  그룹 축소   λ ∈ {1.0(현행), 0.75, 0.50, 0.25, 0.0}
                    out_g = λ·fit_g + (1−λ)·fit_pooled   (λ=0 이면 세 그룹 한 곡선)
  축 S  출발점      1 / 9(현행) / 25 / 49

  전수는 5 × 5 × 4 = 100 칸임. **100 칸을 전부 판정에 쓰면 거짓양성 기계임.**
  (연도짝비교 3개로 보는 게이트의 우연 통과율은 0 이 아니고, PART 4 에서 실측함.)

  그래서 판정 구조를 이렇게 고정함.
    PART 2  **주효과 3개만 판정 대상.** 한 축만 움직이고 나머지는 현행값 고정.
              W: recency vs uniform      L: λ=0.5 vs λ=1.0      S: 25 vs 9
            사전 등록 채택 조건 (§3.6 규칙 6) =
              부호 3/3 일치 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수
    PART 3  100 칸 전체는 **지형 관찰용** 으로만 출력함. 여기서 최대값을 골라 제출하지 않음.
    PART 4  **위약 보정.** 무작위 연도 가중 벡터로 같은 게이트를 N 번 돌려 우연 통과율을
            실측함. PART 2 의 ✅ 는 이 숫자와 나란히 읽어야 함.
    PART 5  주효과에서 통과한 축**만** 결합한 아암 1개. 최종 판정은 LB (§3.6 규칙 4·5).

  그리고 PART 0 이 이 실험 전체의 **상한** 을 먼저 잼. 상한이 작으면 나머지는 볼 필요 없음.

──────────────────────────────────────────────────────────────────────────────
평가 (§3.6 규칙 5 범위 안)
──────────────────────────────────────────────────────────────────────────────
LOYO 중첩 — 연도 y 를 빼고 나머지로 곡선을 적합해 y 에만 적용, y 의 점수를 잼.
§3.6 은 중첩 후처리 추정을 **계측기에서 폐기** 했으나 규칙 5 가 정확히 이 경우를 예외로
둠 — *같은 모델의 같은 OOF 위에서 후처리 방법 A vs B*. 짝비교라 분산이 훨씬 작음.
**단 규칙 5 는 "참고" 까지만 허용함. 최종 판정은 LB.** 이 스크립트는 제출 후보를 하나만
내놓지, 제출 결정을 내리지 않음.

⚠ 평가 점수는 **언제나 무가중 대회 점수**(`group_score`)로 잼. 연도 가중은 *적합* 장치이지
   *평가* 장치가 아님. 가중된 점수로 평가하면 가중을 준 쪽이 자동으로 이김.

──────────────────────────────────────────────────────────────────────────────
실행  (학습 0회. GPU 안 씀. 곡선 적합만 수천 번 돎)
    # 전체 — PART 0~5. 대략 10~15분
    python scripts/step27_post_fitting.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv
    # 빠르게 — PART 3 격자 생략 + 위약 15회. 대략 3~5분
    python scripts/step27_post_fitting.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv --fast
    # 출발점 49 까지 보려면 (+5분)
    python scripts/step27_post_fitting.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv --max-starts 49
    # 통과 시 제출 파일
    python scripts/step27_post_fitting.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv \
        --make-submission --weights recency --lam 0.5 --starts 25 \
        --raw-test ./saved_models/v13/raw_test_preds.csv
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
SUB = "-" * 92

# 실제 모듈의 절점을 씀. 하드코딩하지 않음.
KNOTS = np.asarray(getattr(PP, "KNOTS", [0.0, 0.15, 0.35, 0.60, 0.85, 1.00]), dtype=float)
NK = len(KNOTS)

# ------------------------------------------------------------------ 사전 등록 격자
WEIGHTS = {                       # 연도 -> 상대 가중 (적합 표본에만 적용)
    "uniform":        {2022: 1.0, 2023: 1.0, 2024: 1.0},   # 현행
    "recency":        {2022: 1.0, 2023: 2.0, 2024: 3.0},
    "recency_strong": {2022: 1.0, 2023: 2.0, 2024: 4.0},
    "y2023_down":     {2022: 1.0, 2023: 0.5, 2024: 1.0},   # 기후 근거
    "y2023_up":       {2022: 1.0, 2023: 2.0, 2024: 1.0},   # 부호 대조군
}
LAMBDAS = [1.0, 0.75, 0.50, 0.25, 0.0]                     # 1.0 = 현행(그룹 독립)
STARTSETS = {                                              # 9 = 현행
    1:  ([0.0], [0.0]),
    9:  ([-0.04, 0.0, 0.04], [0.0, 0.08, 0.15]),
    25: ([-0.08, -0.04, 0.0, 0.04, 0.08], [0.0, 0.05, 0.08, 0.12, 0.15]),
    49: ([-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12],
         [0.0, 0.03, 0.06, 0.09, 0.12, 0.17, 0.22]),
}
BASE_W, BASE_L, BASE_S = "uniform", 1.0, 9                 # 현행 = 기준 아암
MAIN_EFFECTS = [                                           # PART 2 판정 대상. 이것만.
    ("W", dict(w="recency",  lam=BASE_L, s=BASE_S), "연도 가중 recency(1,2,3)"),
    ("L", dict(w=BASE_W,     lam=0.50,   s=BASE_S), "그룹 축소 λ=0.5 (pooled 로 절반)"),
    ("S", dict(w=BASE_W,     lam=BASE_L, s=25),     "출발점 9 → 25"),
]


# ================================================================== 점수 (가중형)
def wscore_norm(an, pn, w):
    """정규화(용량 나눈) 단위의 가중 대회 점수. an, pn, w 는 **채점행만** 들어옴."""
    e = np.abs(an - pn)
    sw = w.sum()
    if sw <= 0:
        return np.nan
    nmae = float((w * e).sum() / sw)
    price = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
    den = float((w * an).sum() * 4.0)
    ficr = float((w * an * price).sum() / den) if den > 1e-12 else 0.0
    return 0.5 * (1.0 - nmae) + 0.5 * ficr


def curve(pn, out):
    """절점 KNOTS 에서 out 으로 가는 조각선형 사상. 마지막 절점 밖은 선형 외삽."""
    y = np.interp(pn, KNOTS, out)
    hi = pn > KNOTS[-1]
    if np.any(hi):
        sl = (out[-1] - out[-2]) / (KNOTS[-1] - KNOTS[-2])
        y = np.where(hi, out[-1] + sl * (pn - KNOTS[-1]), y)
    return np.clip(y, 0.0, None)


def mono(x):
    return np.maximum.accumulate(np.clip(np.asarray(x, float), 0.0, 1.4))


# ================================================================== 적합기
def fit_curve(pn, an, w, n_starts, maxiter=500):
    """
    가중 목적함수를 최대화하는 out_knots 를 찾음.
    목적함수가 가격 계단 때문에 불연속이라 기울기 기반을 쓸 수 없음 -> Nelder-Mead.
    단조성은 파라미터 제약이 아니라 목적함수 안에서 maximum.accumulate 로 강제함
    (제약 최적화보다 안정적이고, 이 형태는 현행 구현과 동일한 관례임).

    속도 — 목적함수를 수천 번 부르므로 보간 계수를 **한 번만** 계산해 둠.
      i, f 를 미리 잡으면 매 호출은 gather 2회 + 산술뿐임(np.interp 대비 3~4배).
      i 를 마지막 구간으로 clip 해 두면 f > 1 이 자동으로 선형 외삽이 됨 (curve() 와 동일).
    """
    ss, zs = STARTSETS[n_starts]
    i = np.clip(np.searchsorted(KNOTS, pn, side="right") - 1, 0, NK - 2)
    i1 = i + 1
    f = (pn - KNOTS[i]) / (KNOTS[i1] - KNOTS[i])
    om = 1.0 - f
    sw = float(w.sum())
    wa = w * an
    wa4 = float(wa.sum()) * 4.0

    def neg(x):
        out = mono(x)
        y = np.maximum(out[i] * om + out[i1] * f, 0.0)
        e = np.abs(an - y)
        nmae = float((w * e).sum() / sw)
        price = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
        ficr = float((wa * price).sum() / wa4) if wa4 > 1e-12 else 0.0
        return -(0.5 * (1.0 - nmae) + 0.5 * ficr)

    best_x, best_v = None, -np.inf
    for s, z in itertools.product(ss, zs):
        x0 = KNOTS + s
        x0[0] = z
        x0 = mono(x0)
        r = minimize(neg, x0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, maxfev=maxiter * 2,
                                  xatol=2e-4, fatol=1e-7))
        v = -float(r.fun)
        if np.isfinite(v) and v > best_v:
            best_v, best_x = v, mono(r.x)
    return best_x, best_v


# ================================================================== 데이터 묶음
def build_panel(df, answer, oof, targets, years):
    """
    (year, group) -> dict(pn, an, w0)  — 채점행(a >= 0.10cap)만, 정규화 단위.
    w0 는 연도 가중 이전의 행 가중(전부 1). 나중에 연도 가중을 곱해서 씀.
    """
    panel = {}
    for g in targets:
        cap = CAPACITY_KWH[g]
        a_all = answer[g].to_numpy(float) / cap
        p_all = oof[g].to_numpy(float) / cap
        for y in years:
            m = (df["_year"].to_numpy() == y) & np.isfinite(a_all) & np.isfinite(p_all)
            m &= (a_all >= 0.10)
            if m.sum() < 200:              # G3 처럼 라벨이 없는 (연도, 그룹) 은 제외
                continue
            panel[(y, g)] = dict(pn=p_all[m], an=a_all[m], n=int(m.sum()))
    return panel


def stack_fit_rows(panel, fit_years, groups, wmap):
    """적합 표본을 그룹별/통합으로 쌓음. 연도 가중을 여기서 곱함."""
    per, allp, alla, allw = {}, [], [], []
    for g in groups:
        P, A, W = [], [], []
        for y in fit_years:
            d = panel.get((y, g))
            if d is None:
                continue
            P.append(d["pn"]); A.append(d["an"])
            W.append(np.full(d["n"], float(wmap[y])))
        if not P:
            per[g] = None
            continue
        p, a, w = np.concatenate(P), np.concatenate(A), np.concatenate(W)
        per[g] = (p, a, w)
        # 통합 적합에서는 그룹 크기 차이가 곡선을 큰 그룹 쪽으로 끌지 않도록
        # 그룹별 총가중을 1 로 맞춤 (§3.7 의 '그룹은 대등한 개체' 관례와 일치)
        allp.append(p); alla.append(a); allw.append(w / w.sum())
    pool = (np.concatenate(allp), np.concatenate(alla), np.concatenate(allw)) if allp else None
    return per, pool


# ================================================================== LOYO 중첩 평가
def loyo_eval(panel, answer, df, targets, years, wname, lam, n_starts,
              maxiter=500, cache=None, return_curves=False):
    """
    반환: scores[(year, group)] = 무가중 대회 점수 (연도 y 는 적합에서 제외됨)

    캐시 구조가 이 스크립트의 실행시간을 좌우함.
      * 키 = (wname, n_starts, fold). **λ 는 키에 없음** — λ 는 이미 적합된 두 곡선의
        사후 결합이라 λ 격자 전체가 재적합 0회로 돌아감 (5칸이 공짜).
      * 통합(pooled) 적합은 **λ < 1 일 때만** 함. 전체 적합 횟수의 1/4 을 아낌.
    """
    wmap = WEIGHTS[wname]
    scores, curves = {}, {}
    cache = {} if cache is None else cache
    for y in years:
        fit_years = [t for t in years if t != y]
        key = (wname, n_starts, y)
        if key in cache:
            fits = cache[key]
        else:
            per, _ = stack_fit_rows(panel, fit_years, targets, wmap)
            fits = {g: (fit_curve(*per[g], n_starts, maxiter)[0] if per[g] is not None else None)
                    for g in targets}
            cache[key] = fits

        need_pool = (lam < 1.0) or any(v is None for v in fits.values())
        pool_fit = None
        if need_pool:
            pkey = ("pool",) + key
            if pkey not in cache:
                _, pool = stack_fit_rows(panel, fit_years, targets, wmap)
                cache[pkey] = (fit_curve(*pool, n_starts, maxiter)[0]
                               if pool is not None else None)
            pool_fit = cache[pkey]

        for g in targets:
            d = panel.get((y, g))
            if d is None:
                continue
            fg = fits.get(g)
            if fg is None:
                out = pool_fit                      # 그 폴드에 라벨이 없는 그룹 -> 통합 곡선
            elif lam >= 1.0 or pool_fit is None:
                out = fg
            else:
                out = mono(lam * fg + (1.0 - lam) * pool_fit)
            if out is None:
                continue
            curves[(y, g)] = out
            cap = CAPACITY_KWH[g]
            # 평가는 **무가중 대회 점수**. 채점행 필터는 group_score 가 스스로 함.
            m = (df["_year"].to_numpy() == y)
            a = answer[g].to_numpy(float)[m]
            p = curve(oof_cache[g][m] / cap, out) * cap
            ok = np.isfinite(a) & np.isfinite(p)
            if ok.sum() < 200:
                continue
            scores[(y, g)] = float(group_score(a[ok], np.clip(p[ok], 0, cap), cap)[0])
    return (scores, curves) if return_curves else scores


def raw_scores(answer, df, targets, years):
    """후처리 없음(raw) 연도·그룹 점수 — 상한 계산의 바닥."""
    out = {}
    for g in targets:
        cap = CAPACITY_KWH[g]
        for y in years:
            m = (df["_year"].to_numpy() == y)
            a = answer[g].to_numpy(float)[m]
            p = oof_cache[g][m]
            ok = np.isfinite(a) & np.isfinite(p)
            if ok.sum() < 200:
                continue
            out[(y, g)] = float(group_score(a[ok], np.clip(p[ok], 0, cap), cap)[0])
    return out


# ================================================================== 판정
def yearly_totals(scores, targets, years):
    """연도별 총점 = 그 연도에 존재하는 그룹 점수의 평균 (§3.7: 그룹은 대등)."""
    out = {}
    for y in years:
        v = [scores[(y, g)] for g in targets if (y, g) in scores]
        if v:
            out[y] = float(np.mean(v))
    return out


def judge(sA, sB, targets, years, label_a, label_b, verbose=True):
    """§3.6 규칙 6: 부호일치 ∧ 양수 ∧ |평균|>표준편차 ∧ 그룹 2개 이상 양수."""
    ta, tb = yearly_totals(sA, targets, years), yearly_totals(sB, targets, years)
    ys = [y for y in years if y in ta and y in tb]
    d = np.array([ta[y] - tb[y] for y in ys])
    if len(d) == 0:
        return dict(ok=False, mean=np.nan, std=np.nan, d={}, gpos=0)
    mean, std = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    same = bool(np.all(d > 0) or np.all(d < 0))
    gpos, gline = 0, []
    for g in targets:
        gd = [sA[(y, g)] - sB[(y, g)] for y in years if (y, g) in sA and (y, g) in sB]
        if not gd:
            continue
        gm = float(np.mean(gd))
        gpos += int(gm > 0)
        gline.append(f"{g.replace('kpx_group_', 'G')} {gm:+.4f}")
    ok = bool(same and mean > 0 and abs(mean) > std and gpos >= 2)
    if verbose:
        print(f"    연도별 Δ  " + "  ".join(f"{y} {v:+.4f}" for y, v in zip(ys, d)))
        print(f"    평균 {mean:+.4f}  표준편차 {std:.4f}  신뢰비율 "
              f"{(abs(mean)/std if std > 1e-12 else np.inf):5.2f}   그룹별: " + "  ".join(gline))
        print(f"    => {'✅ 채택 후보' if ok else '❌ 채택 안 함'}"
              f"   (부호일치 {same} · 양수 {mean > 0} · |평균|>표준편차 "
              f"{abs(mean) > std} · 양수그룹 {gpos}/{len(targets)})")
    return dict(ok=ok, mean=mean, std=std, d=dict(zip(ys, d)), gpos=gpos)


# ================================================================== main
oof_cache = {}   # group -> 원단위 예측 ndarray (전 행). loyo_eval 에서 씀.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oof", default="./saved_models/v13/oof_preds.csv")
    ap.add_argument("--fast", action="store_true", help="PART 3 격자 생략, 위약 15회로 축소")
    ap.add_argument("--max-starts", type=int, default=25,
                    help="PART 3 격자에서 돌릴 출발점 상한. 49 를 보려면 49 로 줄 것 (느림)")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--placebo", type=int, default=40, help="PART 4 위약 반복 수")
    ap.add_argument("--seed", type=int, default=20260726)
    # 제출
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--weights", default="recency")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--starts", type=int, default=9)
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
    years = sorted(int(y) for y in df["_year"].unique())

    oof = pd.read_csv(args.oof, index_col=0).reindex(index=df.index, columns=targets).astype(float)
    for g in targets:
        oof_cache[g] = oof[g].to_numpy(float)

    print(BAR)
    print(f"step27 — 후처리 적합 절차 격자탐색")
    print(f"  OOF     {args.oof}")
    print(f"  절점    KNOTS = {np.round(KNOTS, 3).tolist()}  ({NK}개 파라미터/그룹)")
    s = total_score(answer, oof, targets)
    print(f"  raw OOF {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}")
    print(f"  연도    {years}")

    panel = build_panel(df, answer, oof, targets, years)

    # 예보 블록이 익일 00시까지 덮으므로 12-31 블록의 꼬리가 다음 해로 넘어감.
    # 그런 '연도' 는 행이 몇 개뿐이라 폴드가 될 수 없음. 여기서 잘라냄.
    dropped = [y for y in years if not any((y, g) in panel for g in targets)]
    years = [y for y in years if y not in dropped]
    if dropped:
        n_drop = int(df["_year"].isin(dropped).sum())
        print(f"  ⚠ 폴드에서 제외한 연도 {dropped} — 행 {n_drop:,}개뿐임 "
              f"(예보 블록이 익일 00시까지 덮어서 연말 꼬리가 넘어간 것). 정상임.")
    if len(years) < 2:
        print("  ⛔ 폴드가 될 연도가 2개 미만임. LOYO 중첩 비교 불가."); return

    print(f"  채점행  " + "  ".join(
        f"{y}·{g.replace('kpx_group_','G')} {panel[(y,g)]['n']:,}"
        for y in years for g in targets if (y, g) in panel))
    missing = [(y, g) for y in years for g in targets if (y, g) not in panel]
    if missing:
        print(f"  ⚠ 라벨 없음(적합·평가 모두 제외): "
              f"{', '.join(f'{y}·' + g.replace('kpx_group_','G') for y, g in missing)}")

    # 사전 등록 가중표에 없는 연도가 남아 있으면 1.0 으로 둠 (§ 가중은 적합 장치일 뿐).
    for wn, wm in WEIGHTS.items():
        for y in years:
            wm.setdefault(y, 1.0)

    cache = {}
    t0 = time.time()

    # ---------------------------------------------------------------- PART 0
    print(BAR)
    print("PART 0 — 이 실험 전체의 **상한**. 오라클(평가 연도로 직접 적합) − 현행 중첩")
    print("  오라클은 규칙 위반이라 제출 불가지만, **어떤 개선된 적합 절차도 이보다 위로 못 감.**")
    print("  상한이 노이즈 바닥(≈0.001) 수준이면 PART 2~5 는 볼 필요 없이 축 종결임.")
    base = loyo_eval(panel, answer, df, targets, years, BASE_W, BASE_L, BASE_S,
                     args.maxiter, cache)
    rs = raw_scores(answer, df, targets, years)
    # 오라클: 각 (연도, 그룹) 을 그 자신으로 적합
    orc = {}
    for (y, g), d in panel.items():
        w = np.ones(d["n"])
        out, _ = fit_curve(d["pn"], d["an"], w, BASE_S, args.maxiter)
        cap = CAPACITY_KWH[g]
        m = (df["_year"].to_numpy() == y)
        a = answer[g].to_numpy(float)[m]
        p = curve(oof_cache[g][m] / cap, out) * cap
        ok = np.isfinite(a) & np.isfinite(p)
        orc[(y, g)] = float(group_score(a[ok], np.clip(p[ok], 0, cap), cap)[0])

    tr, tb, to = (yearly_totals(rs, targets, years), yearly_totals(base, targets, years),
                  yearly_totals(orc, targets, years))
    yshow = [y for y in years if y in tr and y in tb and y in to]
    print(f"\n  {'연도':<8}{'raw':>10}{'현행 중첩':>12}{'오라클':>10}{'후처리 이득':>12}{'적합 여지':>12}")
    for y in yshow:
        print(f"  {y:<8}{tr[y]:10.4f}{tb[y]:12.4f}{to[y]:10.4f}"
              f"{tb[y]-tr[y]:+12.4f}{to[y]-tb[y]:+12.4f}")
    gap = float(np.mean([to[y] - tb[y] for y in yshow]))
    gain = float(np.mean([tb[y] - tr[y] for y in yshow]))
    print(f"  {'평균':<8}{np.mean(list(tr.values())):10.4f}{np.mean(list(tb.values())):12.4f}"
          f"{np.mean(list(to.values())):10.4f}{gain:+12.4f}{gap:+12.4f}")
    print(f"\n  **상한 {gap:+.4f}.** 이것도 낙관적임 — 오라클은 평가 연도를 통째로 보므로")
    print(f"  적합 절차 개선으로 도달 가능한 몫은 이보다 한참 아래임.")
    if gap < 0.0015:
        print("  ⚠ 상한이 노이즈 바닥 수준임. PART 2 에서 무엇이 ✅ 로 떠도 **제출하지 말 것.**")
    print(f"  참고: 그룹별 상한  " + "  ".join(
        f"{g.replace('kpx_group_','G')} "
        f"{np.mean([orc[(y,g)]-base[(y,g)] for y in years if (y,g) in orc and (y,g) in base]):+.4f}"
        for g in targets))

    # ---------------------------------------------------------------- PART 1
    print(BAR)
    print("PART 1 — 자체 적합기 검증. 배포 파이프라인(`optimize_postprocessing`)과 일치하는가")
    print("  step27 의 비교는 전부 *자체 적합기 안에서* 이뤄지므로 내부적으로는 유효함.")
    print("  다만 여기서 벌어지면 **결론이 배포 파이프라인으로 전이되지 않음.**")
    try:
        pp_ref = PP.optimize_postprocessing(answer, oof, mode="piecewise", verbose=False)
        ref = PP.apply_postprocessing(oof.copy(), pp_ref)
        s_ref = total_score(answer, ref, targets)[0]
    except Exception as e:
        s_ref, pp_ref = np.nan, None
        print(f"  ⚠ optimize_postprocessing 호출 실패: {e}")
    # 자체 적합기, 전 연도 적합 -> 전 연도 적용 (in-sample. ref 와 같은 조건)
    per, pool = stack_fit_rows(panel, years, targets, WEIGHTS["uniform"])
    mine = oof.copy()
    knot_lines = []
    for g in targets:
        if per[g] is None:
            continue
        out, _ = fit_curve(*per[g], BASE_S, args.maxiter)
        knot_lines.append(f"    {g.replace('kpx_group_','G')} out_knots = "
                          f"[{', '.join(f'{v:.3f}' for v in out)}]")
        cap = CAPACITY_KWH[g]
        mine[g] = np.clip(curve(oof_cache[g] / cap, out) * cap, 0, cap)
    s_mine = total_score(answer, mine, targets)[0]
    print("\n".join(knot_lines))
    print(f"    §3.14 기록된 적합값 참고: out_knots = [0.170, 0.206, 0.331, 0.552, 0.759, 0.988]")
    print(f"\n  배포 구현 in-sample 총점 {s_ref:.4f}   자체 적합기 {s_mine:.4f}   "
          f"차이 {s_mine - s_ref:+.4f}")
    if np.isfinite(s_ref) and abs(s_mine - s_ref) > 0.002:
        print("  🛑 벌어짐. §3.6 은 후처리 파라미터를 '절점 6개 + **임계값**' 으로 적었는데")
        print("     자체 적합기는 절점만 씀. 임계값 파라미터가 빠졌을 가능성이 큼.")
        print("     => PART 2~5 결과는 **자체 적합기 내부 비교로만** 읽고, 제출 전에")
        print("        src/postprocessing.py 를 보고 같은 파라미터화로 맞출 것.")
    else:
        print("  ✅ 일치. 자체 적합기 결론이 배포 파이프라인으로 전이됨.")

    # ---------------------------------------------------------------- PART 2
    print(BAR)
    print("PART 2 — 주효과 (사전 등록. **판정 대상은 이 3개뿐**)")
    print(f"  기준 아암 = 현행 (W={BASE_W}, λ={BASE_L}, 출발점={BASE_S})   "
          f"중첩 총점 {np.mean(list(tb.values())):.4f}")
    main_res = {}
    for axis, arm, desc in MAIN_EFFECTS:
        print(f"\n  [{axis}] {desc}")
        sA = loyo_eval(panel, answer, df, targets, years, arm["w"], arm["lam"], arm["s"],
                       args.maxiter, cache)
        ta = yearly_totals(sA, targets, years)
        print(f"    중첩 총점 {np.mean(list(ta.values())):.4f}")
        main_res[axis] = (judge(sA, base, targets, years, axis, "현행"), arm)

    # 부호 대조군 — y2023_down 과 y2023_up 이 둘 다 좋으면 그 축은 잡음
    print(f"\n  [대조] 기후 근거 가중의 부호 대조군 — down 과 up 이 둘 다 양수면 잡음임")
    ctrl = {}
    for wn in ["y2023_down", "y2023_up"]:
        sC = loyo_eval(panel, answer, df, targets, years, wn, BASE_L, BASE_S,
                       args.maxiter, cache)
        r = judge(sC, base, targets, years, wn, "현행", verbose=False)
        ctrl[wn] = r
        print(f"    {wn:<12} 평균 Δ {r['mean']:+.4f}  표준편차 {r['std']:.4f}  "
              f"양수그룹 {r['gpos']}/{len(targets)}  {'✅' if r['ok'] else '❌'}")
    if ctrl["y2023_down"]["mean"] > 0 and ctrl["y2023_up"]["mean"] > 0:
        print("    🛑 **둘 다 양수.** 2023 가중은 신호가 아니라 잡음임 — 기후 근거는 여기서 폐기.")
    elif ctrl["y2023_down"]["mean"] > 0 > ctrl["y2023_up"]["mean"]:
        print("    부호가 갈림 — 기후 근거 방향과 일치함. 단 크기를 PART 4 와 대조할 것.")
    else:
        print("    기후 근거 방향과 **반대**. 2023 은 오히려 더 봐야 하는 해일 수 있음.")

    # ---------------------------------------------------------------- PART 3
    if not args.fast:
        print(BAR)
        print("PART 3 — 전체 격자 (⚠ **지형 관찰용. 여기서 최대값을 골라 제출하지 않음**)")
        print("  100 칸에서 최대를 고르는 것은 사후선택임. PART 4 가 그 위험을 수치로 보여줌.")
        starts_list = [k for k in STARTSETS if k <= args.max_starts]
        if 49 not in starts_list:
            print("  (출발점 49 는 생략함. 보려면 --max-starts 49. 대략 5분 더 걸림)")
        base_mean = np.mean(list(tb.values()))
        for ns in sorted(starts_list):
            print(f"\n  출발점 {ns}개" + ("  ← 현행" if ns == BASE_S else ""))
            print("    " + f"{'가중':<16}" + "".join(f"{('λ=' + f'{l:.2f}'):>10}" for l in LAMBDAS))
            for wn in WEIGHTS:
                row = []
                for lam in LAMBDAS:
                    sX = loyo_eval(panel, answer, df, targets, years, wn, lam, ns,
                                   args.maxiter, cache)
                    row.append(np.mean(list(yearly_totals(sX, targets, years).values())) - base_mean)
                mark = "  ← 현행" if (wn == BASE_W and ns == BASE_S) else ""
                print("    " + f"{wn:<16}" + "".join(f"{v:+10.4f}" for v in row) + mark)
        print(f"\n  (값은 현행 대비 Δ. `uniform × λ=1.00 × 출발점 9` 칸은 기준 아암 자기 자신이므로")
        print(f"   **반드시 +0.0000 이어야 함.** 아니면 캐시나 결정성이 깨진 것임. 경과 {time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- PART 4
    print(BAR)
    print(f"PART 4 — 위약 보정. 무작위 연도 가중 {15 if args.fast else args.placebo}개로 같은 게이트를 돌림")
    print("  가중이 아무 의미 없어도 3년 짝비교 게이트는 우연히 통과할 수 있음.")
    print("  PART 2 의 ✅ 는 반드시 이 통과율과 나란히 읽어야 함.")
    rng = np.random.default_rng(args.seed)
    n_pl = 15 if args.fast else args.placebo
    npass, deltas = 0, []
    for i in range(n_pl):
        wv = rng.uniform(0.5, 3.0, size=len(years))
        wn = f"_pl{i}"
        WEIGHTS[wn] = {y: float(v) for y, v in zip(years, wv)}
        sP = loyo_eval(panel, answer, df, targets, years, wn, BASE_L, BASE_S,
                       args.maxiter, cache)
        r = judge(sP, base, targets, years, wn, "현행", verbose=False)
        npass += int(r["ok"])
        deltas.append(r["mean"])
        del WEIGHTS[wn]
    deltas = np.array([d for d in deltas if np.isfinite(d)])
    print(f"\n  우연 통과 {npass}/{n_pl} = **{npass/max(n_pl,1)*100:.0f}%**")
    print(f"  위약 Δ 분포: 평균 {deltas.mean():+.4f}  표준편차 {deltas.std(ddof=1):.4f}  "
          f"최대 {deltas.max():+.4f}  95분위 {np.quantile(deltas, 0.95):+.4f}")
    print(f"  => **연도 가중 축에서 {np.quantile(deltas, 0.95):+.4f} 이하의 Δ 는 근거가 못 됨.**")
    for axis, (r, _) in main_res.items():
        if axis == "W":
            verdict = ("여전히 유효" if r["mean"] > np.quantile(deltas, 0.95)
                       else "**위약 잡음 범위 안. 폐기**")
            print(f"     PART 2 [W] 의 Δ {r['mean']:+.4f} => {verdict}")

    # ---------------------------------------------------------------- PART 5
    print(BAR)
    print("PART 5 — 결합 아암 (주효과에서 통과한 축만)")
    passed = {a: arm for a, (r, arm) in main_res.items() if r["ok"]}
    if not passed:
        print("  통과 축 없음. **후처리 적합 절차 축 종결.**")
        print("  §3.14 의 '후처리 축은 닫혔다' 를 함수 공간에서 추정량까지 확장해 기록할 것.")
        combo = None
    else:
        combo = dict(w=BASE_W, lam=BASE_L, s=BASE_S)
        for a, arm in passed.items():
            if a == "W": combo["w"] = arm["w"]
            if a == "L": combo["lam"] = arm["lam"]
            if a == "S": combo["s"] = arm["s"]
        print(f"  통과 축: {', '.join(passed)}  =>  결합 아암 "
              f"W={combo['w']} λ={combo['lam']} 출발점={combo['s']}")
        sK = loyo_eval(panel, answer, df, targets, years, combo["w"], combo["lam"], combo["s"],
                       args.maxiter, cache)
        print(f"  중첩 총점 {np.mean(list(yearly_totals(sK, targets, years).values())):.4f}")
        judge(sK, base, targets, years, "결합", "현행")
        print(f"\n  ⚠ §3.6 규칙 5 — 중첩 후처리 비교는 **참고까지**임. 최종 판정은 LB.")
        print(f"     그리고 PART 0 의 상한 {gap:+.4f} 과 PART 4 의 위약 95분위 "
              f"{np.quantile(deltas, 0.95):+.4f} 을 넘지 못하면 제출 가치 없음.")
        print(f"\n  제출: python scripts/step27_post_fitting.py --config {args.config} "
              f"--oof {args.oof} \\\n          --make-submission --weights {combo['w']} "
              f"--lam {combo['lam']} --starts {combo['s']}")

    print(BAR)
    print(f"경과 {time.time()-t0:.0f}s")

    # ---------------------------------------------------------------- 제출
    if args.make_submission:
        print(BAR)
        print(f"제출 파일 생성 — W={args.weights} λ={args.lam} 출발점={args.starts}")
        rt = Path(args.raw_test)
        if not rt.exists():
            print(f"  ⛔ {rt} 없음. raw test 예측이 있어야 후처리를 1회만 적용할 수 있음."); return
        per, pool = stack_fit_rows(panel, years, targets, WEIGHTS[args.weights])
        pool_fit = fit_curve(*pool, args.starts, args.maxiter)[0]
        sub = pd.read_csv(rt)
        for g in targets:
            cap = CAPACITY_KWH[g]
            fg = fit_curve(*per[g], args.starts, args.maxiter)[0] if per[g] is not None else None
            out = pool_fit if fg is None else mono(args.lam * fg + (1 - args.lam) * pool_fit)
            print(f"  {g.replace('kpx_group_','G')} out_knots = "
                  f"[{', '.join(f'{v:.3f}' for v in out)}]")
            sub[g] = np.clip(curve(sub[g].to_numpy(float) / cap, out) * cap, 0, cap)
        sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        outdir = Path(cfg["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
        name = args.out_name or f"submit_pf_{args.weights}_l{args.lam:.2f}_s{args.starts}.csv"
        cols = [c for c in ["forecast_id", "forecast_kst_dtm"] if c in sub.columns] + targets
        sub[cols].to_csv(outdir / name, index=False, encoding="utf-8-sig")
        print(f"\n  🚀 {outdir / name}   행 {len(sub):,}  결측 {int(sub[targets].isna().sum().sum())}")
        for g in targets:
            v = sub[g].to_numpy(float)
            print(f"     {g}: 평균 {np.mean(v)/CAPACITY_KWH[g]*100:5.1f}%cap  "
                  f"최대 {np.max(v)/CAPACITY_KWH[g]*100:5.1f}%cap")


if __name__ == "__main__":
    main()