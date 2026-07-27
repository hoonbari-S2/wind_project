"""
scripts/step28_ficr_geometry.py
==============================================================================
1위와의 격차는 어디에 있는가 — 오차 분포의 기하학과 정확도→FICR 전환율

──────────────────────────────────────────────────────────────────────────────
왜 이걸 먼저 보나
──────────────────────────────────────────────────────────────────────────────
격차를 산식대로 쪼개면 이렇게 됨. (1위 총점 0.672 가정, §3.13 의 1−NMAE 0.8794)

    우리 v13   total 0.641458   1−NMAE 0.867673   FICR 0.415244
    1위        total ~0.672     1−NMAE 0.8794     FICR ~0.4646  (= 2T − (1−NMAE))

    격차 0.0305 = 정확도 몫 0.0059 (**19%**) + FICR 몫 0.0247 (**81%**)

**격차의 5분의 4가 FICR임.** 그런데 여기에 함정이 있음 — FICR 과 NMAE 는 독립이 아님.
정확해지면 자동으로 더 많은 행이 ±6% 안에 들어오므로 FICR 도 같이 오름.
따라서 진짜 물어야 할 것은 이것임.

    **1위의 FICR 우위 중 얼마가 '그냥 더 정확해서' 이고, 얼마가 '오차 분포 모양' 인가?**

전자면 할 일은 정확도 개선(트랙 A)뿐이고, 후자면 **점예측을 밴드에 맞춰 배치하는**
완전히 다른 축이 열려 있다는 뜻임. 지금까지 우리가 한 번도 안 건드린 축임.
이 스크립트는 학습 0회·제출 0회로 그 답을 냄.

──────────────────────────────────────────────────────────────────────────────
PART 1  오차 분포의 지형 — 문턱 근처에 얼마나 쌓여 있나
PART 1b FICR 손실의 지도 — 그룹 × 발전량 분위별로 어디서 돈이 새는가
        (FICR 분모가 Σa·4 이므로 **고발전 행이 압도적으로 비쌈.**
         §3.20 의 'G3 는 고풍속에서 무너진다' 는 FICR 관점에서 행 수보다 훨씬 비싼 손실임)
PART 2  정확도 → FICR 전환곡선. 오차를 κ 배로 줄였을 때 (1−NMAE, FICR) 궤적
        κ 축소는 실현 가능한 방법이 아니라 **'균일하게 더 정확해지면' 의 귀무모형**임.
        여기서 dFICR/d(1−NMAE) 기울기를 뽑음.
PART 3  LB 격차 분해 + **모양 프리미엄**
          모양 프리미엄 = 1위의 ΔFICR − (전환율 × 1위의 Δ정확도)
        0 에 가까우면 1위는 그냥 정확한 것이고, 크면 밴드 배치를 하고 있는 것임.
PART 4  밴드 이동 잠재량 — 6~8% / 8~10% 구간 행을 6% 안으로 넣는 데 필요한 이동량 분포
PART 5  근접 실패 행의 방향성. **예측값 p 로 조건을 검** (실제값 a 로 걸면 §3.16 의
        회귀-평균 인공물이 나옴). p 조건부 편향이 남아 있으면 후처리로는 못 잡는 것임 —
        후처리는 정의상 p 의 단조함수라 p 조건부 편향을 이미 0 으로 만들었어야 하므로,
        **남아 있다면 그건 단조성 제약 때문**이고 다른 축이 필요하다는 신호임.

실행
    python scripts/step28_ficr_geometry.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv
    # 1위 점수를 리더보드에서 보고 넣으면 PART 3 가 정확해짐
    python scripts/step28_ficr_geometry.py --config configs/config_v13.yaml \
        --oof ./saved_models/v13/oof_preds.csv --rank1-total 0.66912 --rank1-nmae 0.88483
==============================================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import CAPACITY_KWH
from src.validation import quiet_warnings, add_time_keys, group_score
import src.postprocessing as PP

quiet_warnings()
BAR = "=" * 92

# 우리 v13 의 실제 LB (§9). 전환율을 LB 좌표로 옮길 때 씀.
US_LB = dict(total=0.641458, nmae=0.867673, ficr=0.415244)

# 2026-07-27 12:00 KST 무렵 public LB 스냅샷. 전 행이 total = 0.5·(1−NMAE)+0.5·FICR 검산 통과.
# (rank, name, total, 1-NMAE, FICR, n_submit)
LB_SNAPSHOT = [
    (1,  "연식2",          0.66912, 0.88483, 0.45342, 62),
    (2,  "별하솜",         0.66888, 0.88787, 0.44989, 32),
    (3,  "GBSNU",          0.66703, 0.87932, 0.45473, 23),
    (4,  "jason99",        0.66529, 0.88587, 0.44470, 59),
    (5,  "코난2",          0.65892, 0.87502, 0.44282, 60),
    (6,  "theseung",       0.65876, 0.87164, 0.44588, 28),
    (7,  "kolonist26",     0.65843, 0.87843, 0.43842, 35),
    (8,  "present01",      0.65841, 0.87484, 0.44198, 8),
    (9,  "기면수",         0.65795, 0.87611, 0.43978, 60),
    (10, "유채원",         0.65781, 0.87597, 0.43965, 56),
    (19, "두바이쫀뜩쿠키", 0.65283, 0.86871, 0.43695, 53),
    (28, "채대언",         0.65057, 0.86367, 0.43746, 43),
    (30, "RIS",            0.65032, 0.87642, 0.42421, 39),
    (40, "chaboom",        0.64869, 0.86878, 0.42860, 32),
]



def gk(g):
    return "G" + g.split("_")[-1]


def ficr_parts(a, e):
    """a = 실제(정규화), e = |오차|(정규화). 생산량 가중 밴드 점유율."""
    w = a / a.sum()
    return (float(w[e <= 0.06].sum()), float(w[(e > 0.06) & (e <= 0.08)].sum()))


def score_from(a, e):
    """정규화 단위의 (1−NMAE, FICR, total). a·e 는 채점행만."""
    nmae = float(e.mean())
    price = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
    ficr = float((a * price).sum() / (a.sum() * 4.0))
    return (1 - nmae, ficr, 0.5 * (1 - nmae) + 0.5 * ficr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oof", default="./saved_models/v13/oof_preds.csv")
    ap.add_argument("--post", choices=["insample", "none"], default="insample",
                    help="LB 숫자는 후처리 후이므로 기본은 후처리 적용(전 연도 in-sample)")
    ap.add_argument("--rank1-total", type=float, default=0.66912)   # 2026-07-27 실측 (연식2)
    ap.add_argument("--rank1-nmae", type=float, default=0.88483)   # 2026-07-27 실측
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()
    oof = pd.read_csv(args.oof, index_col=0).reindex(index=df.index, columns=targets).astype(float)

    if args.post == "insample":
        pp = PP.optimize_postprocessing(answer, oof, mode="piecewise", verbose=False)
        oof = PP.apply_postprocessing(oof.copy(), pp)

    # 채점행만 정규화 단위로 모음
    A, E, P, G = [], [], [], []
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float) / cap
        p = np.clip(oof[g].to_numpy(float) / cap, 0, 1.2)
        m = np.isfinite(a) & np.isfinite(p) & (a >= 0.10)
        A.append(a[m]); P.append(p[m]); E.append(np.abs(a[m] - p[m])); G.append([g] * int(m.sum()))
    a_all = np.concatenate(A); p_all = np.concatenate(P); e_all = np.concatenate(E)
    g_all = np.concatenate(G)

    print(BAR)
    print(f"step28 — 오차 분포의 기하학   (후처리 {args.post}, 채점행 {len(a_all):,})")
    n0, f0, t0 = score_from(a_all, e_all)
    print(f"  OOF 기준선   1−NMAE {n0:.4f}   FICR {f0:.4f}   total {t0:.4f}")
    print(f"  (참고 실제 LB: 1−NMAE {US_LB['nmae']:.4f}  FICR {US_LB['ficr']:.4f}  "
          f"total {US_LB['total']:.4f})")

    # ---------------------------------------------------------------- PART 1
    print(BAR)
    print("PART 1 — 오차 분포의 지형. **생산량 가중** 질량이 문턱 어디에 쌓여 있나")
    edges = [0, .02, .04, .06, .08, .10, .15, .25, 9]
    print(f"  {'|오차| 구간':<16}{'행 비중':>10}{'생산량 비중':>12}{'누적 생산량':>12}   단가")
    wa = a_all / a_all.sum()
    cum = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e_all >= lo) & (e_all < hi)
        cum += wa[m].sum()
        price = "4.0원" if hi <= 0.06 else ("3.0원" if hi <= 0.08 else "0")
        lab = f"{lo*100:.0f}~{hi*100:.0f}%cap" if hi < 9 else f"{lo*100:.0f}%cap 이상"
        print(f"  {lab:<16}{m.mean()*100:9.1f}%{wa[m].sum()*100:11.1f}%{cum*100:11.1f}%   {price}")
    w6, w68 = ficr_parts(a_all, e_all)
    print(f"\n  6% 안 {w6*100:.1f}%   6~8% {w68*100:.1f}%   =>  FICR {w6 + 0.75*w68:.4f}")
    m_near = (e_all > 0.06) & (e_all <= 0.10)
    print(f"  **근접 실패(6~10%) 생산량 비중 {wa[m_near].sum()*100:.1f}%** — 이 덩어리가")
    print(f"  전부 6% 안으로 들어오면 FICR {w6 + wa[m_near].sum():.4f} (+{wa[m_near].sum() - 0.25*w68:.4f}).")
    print(f"  1위와의 FICR 격차가 이 덩어리 안에서 해결 가능한 크기인지 PART 3 과 대조할 것.")

    # ---------------------------------------------------------------- PART 1b
    print(BAR)
    print("PART 1b — FICR 손실의 지도. 그룹 × 실제 발전량 5분위")
    print("  FICR 분모가 Σa·4 이므로 **고발전 행 1개가 저발전 행 여러 개만큼 비쌈.**")
    q = np.quantile(a_all, [0, .2, .4, .6, .8, 1.0])
    print(f"\n  {'그룹':<6}" + "".join(f"{'Q'+str(i+1):>13}" for i in range(5)) + f"{'그룹 합':>11}")
    print(f"  {'':<6}" + "".join(f"{'(생산량몫/포획)':>13}" for _ in range(5)))
    for g in targets:
        mg = (g_all == g)
        cells, tot_w, tot_c = [], 0.0, 0.0
        for i in range(5):
            m = mg & (a_all >= q[i]) & (a_all <= q[i + 1] if i == 4 else a_all < q[i + 1])
            if m.sum() == 0:
                cells.append("      -/-   "); continue
            wq = a_all[m].sum() / a_all.sum()
            pr = np.where(e_all[m] <= 0.06, 4.0, np.where(e_all[m] <= 0.08, 3.0, 0.0))
            cap_rate = float((a_all[m] * pr).sum() / (a_all[m].sum() * 4.0))
            tot_w += wq; tot_c += wq * cap_rate
            cells.append(f"{wq*100:5.1f}%/{cap_rate*100:4.0f}%")
        print(f"  {gk(g):<6}" + "".join(f"{c:>13}" for c in cells)
              + f"{tot_w*100:8.1f}%/{(tot_c/max(tot_w,1e-9))*100:.0f}%")
    print("\n  '생산량몫' = 그 칸이 FICR 분모에서 차지하는 비중, '포획' = 그 칸의 FICR 달성률.")
    print("  **생산량몫이 크면서 포획이 낮은 칸이 돈이 새는 곳임.**")

    # ---------------------------------------------------------------- PART 2
    print(BAR)
    print("PART 2 — 정확도 → FICR 전환곡선.  p' = a + κ(p−a) 로 오차만 균일 축소")
    print("  실현 방법이 아니라 **'그냥 더 정확해지면' 의 귀무모형**임.")
    print(f"\n  {'κ':>6}{'1−NMAE':>10}{'FICR':>10}{'total':>10}{'Δ1−NMAE':>11}{'ΔFICR':>10}{'전환율':>9}")
    ks = [1.6, 1.4, 1.2, 1.1, 1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    rows = []
    for k in ks:
        e = e_all * k
        n, f, t = score_from(a_all, e)
        rows.append((k, n, f, t, n - n0, f - f0))
        print(f"  {k:6.2f}{n:10.4f}{f:10.4f}{t:10.4f}{n-n0:+11.4f}{f-f0:+10.4f}"
              f"{((f-f0)/(n-n0) if abs(n-n0) > 1e-9 else 0):9.2f}")

    # **LB 작동점에 맞춘 국소 전환율.**
    #   OOF 의 1−NMAE 는 LB 와 다름(중첩/연도 차이). 전환율은 곡선 위 위치에 따라 달라지므로
    #   κ=1(OOF 작동점)에서 잰 값을 LB 격차에 그대로 쓰면 틀림. LB 의 1−NMAE 와 같아지는
    #   κ 를 찾아 **그 지점의 국소 기울기**를 씀.
    nn = np.array([r[1] for r in rows]); ff = np.array([r[2] for r in rows])
    o = np.argsort(nn)
    nn, ff = nn[o], ff[o]
    tgt = US_LB["nmae"]
    if nn.min() <= tgt <= nn.max():
        j = int(np.clip(np.searchsorted(nn, tgt), 1, len(nn) - 1))
        slope = float((ff[j] - ff[j - 1]) / (nn[j] - nn[j - 1]))
        anchor = f"LB 작동점 1−NMAE={tgt:.4f} 부근 (곡선 위 보간)"
    else:
        slope = float((ff[-1] - ff[-2]) / (nn[-1] - nn[-2]))
        anchor = f"⚠ LB 작동점이 곡선 범위 밖임. 곡선 끝에서 잼 (OOF {n0:.4f} vs LB {tgt:.4f})"
    print(f"\n  **국소 전환율 dFICR/d(1−NMAE) ≈ {slope:.2f}**   기준점: {anchor}")
    print(f"  즉 정확도 1−NMAE 를 +0.0100 올리면 FICR 이 자동으로 +{slope*0.01:.4f} 따라옴.")
    print(f"  총점으로는 0.5·0.0100 + 0.5·{slope*0.01:.4f} = {0.5*0.01 + 0.5*slope*0.01:+.4f}")
    print(f"  => **정확도 1점의 총점 배율은 {(0.5 + 0.5*slope):.2f}배임.**")
    print(f"     §8.3 의 트랙 A 몫(raw 기준)은 이 배율만큼 과소평가돼 있었음.")

    # ---------------------------------------------------------------- PART 3
    print(BAR)
    print("PART 3 — LB 격차 분해와 **모양 프리미엄**")
    t1, n1 = args.rank1_total, args.rank1_nmae
    f1 = 2 * t1 - n1
    dn, dfi = n1 - US_LB["nmae"], f1 - US_LB["ficr"]
    print(f"  우리 LB   total {US_LB['total']:.4f}  1−NMAE {US_LB['nmae']:.4f}  FICR {US_LB['ficr']:.4f}")
    print(f"  1위  LB   total {t1:.4f}  1−NMAE {n1:.4f}  FICR {f1:.4f}  (FICR = 2T − (1−NMAE))")
    print(f"\n  격차 {t1-US_LB['total']:+.4f} = 정확도 {0.5*dn:+.4f} "
          f"({100*0.5*dn/(t1-US_LB['total']):.0f}%) + FICR {0.5*dfi:+.4f} "
          f"({100*0.5*dfi/(t1-US_LB['total']):.0f}%)")
    explained = slope * dn
    premium = dfi - explained
    print(f"\n  정확도 우위 {dn:+.4f} 로 **자동 따라오는** FICR = {slope:.2f} × {dn:+.4f} = {explained:+.4f}")
    print(f"  실제 FICR 우위 {dfi:+.4f}")
    print(f"  **모양 프리미엄 = {premium:+.4f}**   (총점 환산 {0.5*premium:+.4f})")
    print()
    if premium > 0.015:
        print("  => **1위는 정확도만으로 설명 안 됨.** 같은 정확도에서 우리보다 밴드를 더 먹고 있음.")
        print("     오차 분포의 *모양* 이 다르다는 뜻이고, 그건 점예측을 밴드에 맞춰 배치할 때만 생김.")
        print("     후처리(p 의 단조함수)로는 원리적으로 못 만듦 — §3.14 가 증명한 그대로임.")
        print("     => 열려 있는 축은 둘. (1) 밴드 추종 목적함수  (2) 분위회귀 + 행별 기대효용(B-2b)")
    elif premium > 0.005:
        print("  => 모양 프리미엄이 중간 크기임. 정확도와 밴드 배치 **둘 다** 필요함.")
    else:
        print("  => **1위는 그냥 더 정확한 것임.** 밴드 마법 없음.")
        print("     그러면 할 일은 트랙 A(정확도) 하나뿐이고, 밴드 트랙은 §8.2 대로 닫아 두면 됨.")
    print(f"\n  ⚠ 전환율 {slope:.2f} 는 **우리 오차 분포 모양**에서 잰 것임. 1위의 모양이 다르면")
    print(f"     그들의 전환율도 다름. 그래서 이 프리미엄은 정확한 값이 아니라 **부호와 자릿수**로만 읽을 것.")


    # ---------------------------------------------------------------- PART 3b
    print(BAR)
    print("PART 3b — 필드 횡단면 (2026-07-27 public LB 스냅샷, 오프라인 하드코딩)")
    print("  PART 3 은 우리 오차 분포에 기대는 모형 계산이지만, 이건 **관측**임.")
    top10 = [r for r in LB_SNAPSHOT if r[0] <= 10]
    nn = np.array([r[3] for r in top10]); ff = np.array([r[4] for r in top10])
    b1v, b0v = np.polyfit(nn, ff, 1)
    pred = b0v + b1v * US_LB["nmae"]
    resid = US_LB["ficr"] - pred
    print(f"\n  상위 10팀 회귀: FICR = {b0v:+.4f} + {b1v:.3f}×(1−NMAE)   "
          f"(r={np.corrcoef(nn, ff)[0,1]:.2f} — 참고용)")
    print(f"  우리 정확도({US_LB['nmae']:.4f})에서 필드 예측 FICR {pred:.4f} vs 실제 {US_LB['ficr']:.4f}"
          f"   **잔차 {resid:+.4f}** (총점 {0.5*resid:+.4f})")
    print("\n  회귀 가정이 싫으면 이 네 팀만 보면 됨 — **점별 비교라 가정이 없음**:")
    print(f"  {'팀(순위)':<22}{'1−NMAE':>9}{'Δ정확도':>10}{'FICR':>9}{'ΔFICR':>9}")
    for rk in [6, 19, 28, 40]:
        r = next(x for x in LB_SNAPSHOT if x[0] == rk)
        print(f"  {r[1] + f'(#{rk})':<22}{r[3]:9.4f}{r[3]-US_LB['nmae']:+10.4f}"
              f"{r[4]:9.4f}{r[4]-US_LB['ficr']:+9.4f}")
    print("\n  #28 채대언은 **우리보다 부정확한데**(−0.0040) FICR 을 +0.0221 더 가져감.")
    print("  #40 chaboom 은 정확도가 사실상 같은데(+0.0011) FICR +0.0134.")
    print("  => **같은 정확도에서 +0.013~0.022 의 FICR 이 필드에서 실현되고 있음.**")
    print("     우리 단조 후처리는 그 함수 공간의 국소최적(§3.15)이므로 이 몫은 후처리 밖에 있음:")
    print("     (a) 오차 분포 중심이 더 뾰족 (모델·앙상블·손실 모양)")
    print("     (b) x-조건부 점예측 배치 (B-2b)")
    print("     (c) 발전량-가중 배치 (FICR 은 a-가중, NMAE 는 행-균등)")
    print("  ⚠ 이건 방법을 알려주지 않음. **크기와 존재**만 알려줌. PART 3 의 프리미엄과 같이 읽을 것.")

    # ---------------------------------------------------------------- PART 4
    print(BAR)
    print("PART 4 — 밴드 이동 잠재량. 근접 실패 행을 6% 안에 넣으려면 얼마나 옮겨야 하나")
    need = e_all - 0.06                       # 6% 안으로 들어가는 데 필요한 최소 이동 (%cap)
    print(f"  {'필요 이동량':<16}{'행 비중':>10}{'생산량 비중':>12}{'누적 FICR 이득':>16}")
    gain = 0.0
    for lo, hi in [(0, .01), (.01, .02), (.02, .03), (.03, .04), (.04, .06)]:
        m = (need > lo) & (need <= hi)
        add = float((a_all[m] * (4.0 - np.where(e_all[m] <= 0.08, 3.0, 0.0))).sum()
                    / (a_all.sum() * 4.0))
        gain += add
        print(f"  {f'{lo*100:.0f}~{hi*100:.0f}%cap':<16}{m.mean()*100:9.1f}%"
              f"{(a_all[m]/a_all.sum()).sum()*100:11.1f}%{gain:+16.4f}")
    print(f"\n  **2%cap 이내로 옮길 수 있으면 FICR +{gain:.4f} 규모가 열림** (누적 3번째 줄까지).")
    print(f"  2%cap = {0.02*21600:.0f}kWh 임. 지금 NMAE 가 {(1-n0)*100:.1f}%cap 이므로")
    print(f"  '평균 오차보다 작은 이동' 으로 닿는 거리임 — 즉 **정보가 아니라 배치의 문제**일 수 있음.")

    # ---------------------------------------------------------------- PART 5
    print(BAR)
    print("PART 5 — 근접 실패 행의 방향성. **예측값 p 로 조건을 검**")
    print("  실제값 a 로 조건을 걸면 §3.16 의 회귀-평균 인공물이 나옴. p 조건부여야 함.")
    print("  후처리는 p 의 단조함수이므로 p 조건부 편향을 이미 0 으로 만들었어야 함.")
    print("  **그런데도 남아 있다면 그것은 단조성 제약이 만든 잔차**이고, 다른 축이 필요하다는 신호임.\n")
    qp = np.quantile(p_all, np.linspace(0, 1, 7))
    print(f"  {'p 구간(%cap)':<16}{'행수':>8}{'평균 p−a':>11}{'6%포획':>9}{'생산량몫':>10}"
          f"{'과대 비중':>11}")
    for i in range(6):
        m = (p_all >= qp[i]) & (p_all < qp[i + 1] if i < 5 else p_all <= qp[i + 1])
        if m.sum() < 50:
            continue
        d = p_all[m] - a_all[m]
        pr = np.where(e_all[m] <= 0.06, 4.0, np.where(e_all[m] <= 0.08, 3.0, 0.0))
        near = m & (e_all > 0.06) & (e_all <= 0.10)
        over = float((p_all[near] > a_all[near]).mean() * 100) if near.sum() else np.nan
        print(f"  {f'{qp[i]*100:.0f}~{qp[i+1]*100:.0f}':<16}{m.sum():8,}{d.mean()*100:+10.2f}%"
              f"{(pr >= 4).mean()*100:8.0f}%{(a_all[m]/a_all.sum()).sum()*100:9.1f}%"
              f"{over:10.0f}%")
    print("\n  '과대 비중' = 그 p 구간의 근접 실패(6~10%) 행 중 **과대예측** 비율.")
    print("  50% 근처면 대칭이라 단조 이동으로 못 고침. 한쪽으로 크게 치우친 구간이 있으면")
    print("  그 구간은 **후처리가 다른 구간과의 상충 때문에 포기한 곳**이고, 조건부 축이 열림.")
    print(BAR)


if __name__ == "__main__":
    main()