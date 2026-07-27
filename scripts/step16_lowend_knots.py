"""
scripts/step16_lowend_knots.py
==============================================================================
후처리 이득이 사는 곳을 찾았다 — 그리고 거기엔 매듭이 없다

step15 PART 1 표가 준 것 (사전 등록 판정과 무관하게 나온 부수 발견)

    예측수준        S1        S2        S3     기울기
    L1 최저    +0.3492   +0.3259   +0.2593   -0.0899     <- 여기
    L2         +0.0161   +0.0130   +0.0233   +0.0072
    L3 최고    -0.0098   +0.0125   +0.0301   +0.0398

  후처리 이득의 사실상 전부가 **예측 최저 3분위** 에 있다. L2/L3 는 0 근처다.
  (L3 S1 은 음수다 — 확신하는 고출력 행에서는 후처리가 오히려 손해다)

  그런데 현행 절점은
      KNOTS = [0.00, 0.15, 0.35, 0.60, 0.85, 1.00]
  이고 풍력 예측 분포는 0 쪽으로 크게 치우쳐 있어 L1 전체가 **[0.00, 0.15] 한 구간**
  안에 들어간다. 적합된 out_knots 는 [0.17, 0.206, ...] 이므로 그 구간의 사상은
      p in [0, 0.15]  ->  0.17 ~ 0.206
  거의 상수다. **이득의 100% 가 나오는 영역에 자유도가 사실상 0 개다.**

  §8.4 는 '전역 후처리의 매듭 수·형태 튜닝 = 포화' 로 적어 두었다. 그 판단은
  이득이 어디에 몰려 있는지 모르는 상태에서 내려졌다. 이 스크립트가 그 재심이다.

왜 이것이 밴드 트랙의 본체인가
  채점 필터는 a >= 0.10cap 이다. 모델이 낮게 예측했는데 채점 대상인 행 = 컷인 무릎이다.
  P ~ v^3 이라 그 구간은 풍속 오차가 발전량 오차로 가장 크게 증폭된다.
  §1.3 이 증명한 floor 우월성은 정확히 이 행들의 이야기이고, 지금은 그것을
  **전역 상수 하나**로 처리하고 있다. 조건부 값으로 바꾸는 것이 B-2 의 실체다.

설계 — 학습 0회. 저장된 OOF 만 읽는다.
  같은 v13 OOF 위에서 절점 격자만 바꾼다. 나머지 절차(piecewise, zero_th, LOYO 중첩)는
  전부 동일. §3.6 규칙 5 가 허용하는 '같은 OOF 위 후처리 A vs B' 짝비교다.
  최종 판정은 LB.

판정 (사전 등록 — 결과 보기 전에 고정)
  1) 연도별 짝비교에서 부호 3/3 일치 ∧ |평균| > 표준편차   -> 제출 1회
  2) 부호가 갈리면 제출하지 않는다
  3) 이득이 L1 에 몰려 있지 않으면(= 기여 분해에서 L1 몫 < 50%) 기전이 다른 것이므로
     통과했더라도 제출 전에 원인을 먼저 밝힌다

실행
    python scripts/step16_lowend_knots.py --config configs/config_v13.yaml
    python scripts/step16_lowend_knots.py --config configs/config_v13.yaml \
        --knots 0,0.02,0.05,0.09,0.14,0.22,0.35,0.60,0.85,1.0
    python scripts/step16_lowend_knots.py --config configs/config_v13.yaml --make-submission
==============================================================================
"""
import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import CAPACITY_KWH
from src.validation import (quiet_warnings, add_time_keys, total_score, group_score,
                            is_difference_real)
import src.postprocessing as PP
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78
NEW_KNOTS = [0.00, 0.03, 0.06, 0.10, 0.15, 0.25, 0.35, 0.60, 0.85, 1.00]
NL = 3                                  # 기여 분해용 예측 수준 칸 수


@contextmanager
def knots(grid):
    """PP.KNOTS 는 optimize 와 apply 가 함께 참조하는 x축이다. 반드시 짝을 맞춰 바꾼다."""
    old = PP.KNOTS
    PP.KNOTS = np.asarray(grid, dtype=float)
    try:
        yield
    finally:
        PP.KNOTS = old


def nested_post(answer, oof, yrs, years, targets, grid, max_starts=None):
    """연도 하나를 빼고 적합해 그 해에 적용. grid 로 절점을 갈아끼운다."""
    out = pd.DataFrame(index=oof.index, columns=targets, dtype=float)
    with knots(grid):
        for y in years:
            te = (yrs == y)
            tr = ~te & answer.notna().any(axis=1).to_numpy()
            pp = optimize_postprocessing(answer.loc[tr], oof.loc[tr],
                                         mode="piecewise", verbose=False)
            out.loc[te, targets] = apply_postprocessing(oof.loc[te].copy(), pp)[targets].values
    return out


def level_share(answer, oof, post, targets, yrs, years):
    """
    후처리 이득의 예측수준별 기여 분해.
    NMAE 항은 행수 가중으로 정확히 분해되고 FICR 은 발전량 가중이라 근사다.
    '어디서 이득이 났는가' 를 보는 용도로만 쓴다.
    """
    tot_w, tot_d = np.zeros(NL), np.zeros(NL)
    for y in years:
        te = (yrs == y)
        for g in targets:
            cap = CAPACITY_KWH[g]
            a = answer.loc[te, g].to_numpy(float)
            r = oof.loc[te, g].to_numpy(float)
            p = post.loc[te, g].to_numpy(float)
            fin = np.isfinite(a) & np.isfinite(r) & np.isfinite(p) & (a >= 0.10 * cap)
            if fin.sum() < 300:
                continue
            e = np.nanquantile(r[np.isfinite(r)], np.linspace(0, 1, NL + 1))
            e[0], e[-1] = -np.inf, np.inf
            b = np.clip(np.digitize(r, e[1:-1]), 0, NL - 1)
            for k in range(NL):
                m = fin & (b == k)
                if m.sum() < 100:
                    continue
                s_r = group_score(a[m], r[m], cap)[0]
                s_p = group_score(a[m], p[m], cap)[0]
                if np.isfinite(s_r) and np.isfinite(s_p):
                    tot_w[k] += m.sum()
                    tot_d[k] += m.sum() * (s_p - s_r)
    return tot_w, tot_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-base", default="./saved_models/v13")
    ap.add_argument("--knots", default=None, help="쉼표구분 신규 절점. 미지정시 기본 격자")
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--out-name", default="submit_v15_lowknots.csv")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    db = Path(args.dir_base)
    old_grid = np.asarray(PP.KNOTS, float)
    new_grid = (np.array([float(v) for v in args.knots.split(",")])
                if args.knots else np.array(NEW_KNOTS))

    tdir = Path(cfg["data_paths"]["train_dir"])
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()
    yrs = df["_year"].to_numpy()
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]
    ob = pd.read_csv(db / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=targets).astype(float)

    print(BAR)
    s0 = total_score(answer, ob, targets)
    print(f"기준 {db}   raw OOF {s0[0]:.4f}   연도 {years}")
    print(f"  현행 절점 {len(old_grid):2d}개  " + " ".join(f"{v:.2f}" for v in old_grid))
    print(f"  신규 절점 {len(new_grid):2d}개  " + " ".join(f"{v:.2f}" for v in new_grid))

    # ---- 왜 저쪽에 자유도가 필요한지: 채점행 예측 분포 -----------------------
    print(f"\n  [자유도 점검] 채점 대상(a>=10%cap) 행의 예측값이 현행 첫 구간에 몇 % 있는가")
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float); p = ob[g].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(p) & (a >= 0.10 * cap)
        frac = np.mean(p[ok] / cap < old_grid[1]) * 100
        print(f"    {g.replace('kpx_',''):<10s} p/cap < {old_grid[1]:.2f} 인 채점행 {frac:5.1f}%"
              f"   (그 구간의 절점 개수: 2 -> {int((new_grid < old_grid[1]).sum())+1})")

    # ================================================================ 짝비교
    print(BAR)
    print("절점 격자 A(현행) vs B(신규) — 같은 OOF, LOYO 중첩, 나머지 절차 동일")
    post_a = nested_post(answer, ob, yrs, years, targets, old_grid)
    post_b = nested_post(answer, ob, yrs, years, targets, new_grid)

    print(f"\n  {'연도':>6s}{'A 현행':>12s}{'B 신규':>12s}{'Δ':>12s}")
    for y in years:
        te = (yrs == y)
        sa = total_score(answer.loc[te], post_a.loc[te], targets)[0]
        sb = total_score(answer.loc[te], post_b.loc[te], targets)[0]
        print(f"  {y:6d}{sa:12.4f}{sb:12.4f}{sb-sa:+12.4f}")

    print(f"\n  [그룹별 — §3.7: 총점으로 합치면 상쇄가 신호를 지운다]")
    print("    " + f"{'그룹':<12s}{'A 현행':>12s}{'B 신규':>12s}{'Δ':>12s}")
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float)
        sa = group_score(a, post_a[g].to_numpy(float), cap)[0]
        sb = group_score(a, post_b[g].to_numpy(float), cap)[0]
        print("    " + f"{g.replace('kpx_',''):<12s}{sa:12.4f}{sb:12.4f}{sb-sa:+12.4f}")

    print(f"\n  [연도짝비교 판정]")
    res = is_difference_real(df, answer, post_b, post_a, targets,
                             name_a="B 신규절점", name_b="A 현행")

    # ================================================================ 기여 분해
    print(BAR)
    print("이득이 어디서 났는가 — 예측수준 3분위 기여 분해 (NMAE 항은 정확, FICR 항은 근사)")
    for lab, post in [("A 현행", post_a), ("B 신규", post_b)]:
        w, d = level_share(answer, ob, post, targets, yrs, years)
        tot = d.sum()
        print(f"\n  [{lab}]  후처리 총 이득(가중합) {tot/max(w.sum(),1):+.4f}")
        print("    " + f"{'수준':<10s}{'채점행수':>12s}{'행비중':>10s}{'칸 Δ':>12s}{'이득 몫':>10s}")
        for k in range(NL):
            lab_k = f"L{k+1}" + (" 최저" if k == 0 else " 최고" if k == NL - 1 else "")
            share = d[k] / tot * 100 if abs(tot) > 1e-12 else np.nan
            print("    " + f"{lab_k:<10s}{int(w[k]):12d}{w[k]/max(w.sum(),1)*100:9.1f}%"
                  f"{d[k]/max(w[k],1):+12.4f}{share:9.1f}%")

    # ================================================================ 제출
    if args.make_submission:
        print(BAR)
        tb = db / "raw_test_preds.csv"
        if not tb.exists():
            print(f"  ⚠ {tb} 없음. main/inference.py --config {args.config} 를 먼저 돌릴 것.")
        elif not (res and res["real"]):
            print("  ⛔ 연도짝비교가 노이즈 판정이다. 제출 파일을 만들지 않는다.")
        else:
            pb = pd.read_csv(tb)
            with knots(new_grid):
                pp = optimize_postprocessing(answer, ob, mode="piecewise", verbose=False)
                sub = apply_postprocessing(pb.copy(), pp)
                for g in targets:
                    print(f"  {g}: out_knots = ["
                          + " ".join(f"{v:.3f}" for v in pp[g]["out_knots"]) + "]"
                          + f"  zero_th {pp[g]['zero_th']/CAPACITY_KWH[g]:.3f}cap")
            sub["forecast_kst_dtm"] = pd.to_datetime(
                sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            outdir = Path(cfg["data_paths"]["submission_dir"])
            outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / args.out_name
            sub.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  🚀 제출 파일 저장: {out}")

    print(BAR); print(f"⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()