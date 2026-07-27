"""
scripts/premium_track.py
==============================================================================
밴드 프리미엄 추적기

정의
  price(e) = 4 (e<=0.06), 3 (0.06<e<=0.08), 0 (그 외).  e 는 용량 정규화 절대오차.
  FICR = sum(actual*price) / sum(actual*4).

  같은 NMAE 라도 오차가 6%/8% 경계 안쪽에 얼마나 몰려 있느냐에 따라 FICR 이 달라진다.
  '함의 FICR' = 오차가 반정규분포(half-normal)를 따르고 발전량과 독립일 때
                그 NMAE 가 자연히 만들어내는 FICR.
  '프리미엄'  = 실제 FICR - 함의 FICR.  즉 밴드를 얼마나 잘 활용했는가.

  이 분해가 필요한 이유: 총점만 보면 상대가 정확도로 이겼는지 밴드로 이겼는지 알 수 없다.
  실제로 현재 1위와 2위는 정반대 전략이다 (§3.13).

검증
  기록된 프리미엄과 대조했을 때 v13 +0.0668(기록 +0.0671), 구1위 +0.0519(기록 +0.0523).
  반정규 가정이 이 대회 오차분포를 잘 근사한다.

실행
  python scripts/premium_track.py                       # 내장 기록 + 리더보드
  python scripts/premium_track.py --log experiment_log.xlsx   # 로그 전체를 훑는다
  python scripts/premium_track.py --add 0.8794 0.4653 "1위(신)"
==============================================================================
"""
import argparse
from pathlib import Path

import numpy as np

try:
    from scipy.stats import norm
    _cdf = norm.cdf
except ImportError:                       # scipy 없으면 erf 로 대체
    from math import erf, sqrt
    _cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))

BAR = "=" * 82

# 참조점 (수기 관리). 리더보드 상위는 여기서 갱신한다.
REFS = [
    ("v6",         0.863924, 0.393937),
    ("v11",        0.864890, 0.407743),
    ("v12",        0.864932, 0.408809),
    ("v13",        0.867673, 0.415244),
    ("v14",        0.862562, 0.396389),
    ("v14-nopost", 0.861542, 0.392183),
    ("2위(구1위)",   0.886900, 0.454500),
    ("1위(신)",     0.879360, 0.465270),
]


def implied_ficr(nmae):
    """오차 ~ half-normal(평균=nmae), 발전량과 독립 가정에서의 FICR."""
    nmae = np.asarray(nmae, float)
    s = nmae * np.sqrt(np.pi / 2)
    p6 = 2 * _cdf(0.06 / s) - 1
    p8 = 2 * _cdf(0.08 / s) - 1
    return p6 + 0.75 * (p8 - p6)


def report(rows):
    print(BAR)
    print(f"{'':14s}{'1-nMAE':>10s}{'FICR':>10s}{'함의FICR':>11s}{'프리미엄':>11s}{'총점':>11s}")
    out = {}
    for name, a, f in rows:
        if not (np.isfinite(a) and np.isfinite(f)):
            continue
        imp = float(implied_ficr(1 - a))
        out[name] = (a, f, f - imp)
        print(f"{name:14s}{a:10.4f}{f:10.4f}{imp:11.4f}{f-imp:+11.4f}{0.5*a+0.5*f:11.6f}")
    return out


def compare(out, mine, top):
    if mine not in out or top not in out:
        return
    a0, f0, p0 = out[mine]
    a1, f1, p1 = out[top]
    s0, s1 = 0.5 * a0 + 0.5 * f0, 0.5 * a1 + 0.5 * f1
    print(BAR)
    print(f"갭 분해  ({mine} -> {top})   총 갭 {s1-s0:+.6f}")
    # 반사실 1: 정확도만 따라잡고 프리미엄은 우리 것 유지
    f_acc = float(implied_ficr(1 - a1)) + p0
    s_acc = 0.5 * a1 + 0.5 * f_acc
    # 반사실 2: 프리미엄만 따라잡고 정확도는 우리 것 유지
    f_prm = float(implied_ficr(1 - a0)) + p1
    s_prm = 0.5 * a0 + 0.5 * f_prm
    print(f"  정확도만 같아지면   {s_acc:.6f}   (남는 갭 {s1-s_acc:+.6f})  <- 밴드가 가진 몫")
    print(f"  프리미엄만 같아지면 {s_prm:.6f}   (남는 갭 {s1-s_prm:+.6f})  <- 정확도가 가진 몫")
    tot = (s1 - s_acc) + (s1 - s_prm)
    if tot > 0:
        # s1-s_acc = 정확도를 따라잡고도 남는 갭 = 밴드가 가진 몫
        # s1-s_prm = 프리미엄을 따라잡고도 남는 갭 = 정확도가 가진 몫
        print(f"  대략 비중          정확도 {100*(s1-s_prm)/tot:.0f}%  /  밴드 {100*(s1-s_acc)/tot:.0f}%")
    print(f"  프리미엄 차이       {p1-p0:+.4f}   (FICR 기준. 총점 기준으로는 {0.5*(p1-p0):+.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="experiment_log.xlsx 를 읽어 LB 기록도 포함")
    ap.add_argument("--add", nargs=3, action="append", metavar=("1-nMAE", "FICR", "NAME"),
                    help="참조점 추가")
    ap.add_argument("--mine", default="v13")
    ap.add_argument("--top", default="1위(신)")
    args = ap.parse_args()

    rows = list(REFS)
    if args.log and Path(args.log).exists():
        import pandas as pd
        d = pd.read_excel(args.log)
        need = {"Public_LB_1-nMAE", "Public_LB_FiCR", "Version"}
        if need <= set(d.columns):
            known = {n for n, _, _ in rows}
            for _, r in d.iterrows():
                v, a, f = r["Version"], r["Public_LB_1-nMAE"], r["Public_LB_FiCR"]
                if pd.notna(v) and pd.notna(a) and pd.notna(f) and str(v) not in known:
                    rows.append((str(v), float(a), float(f)))
        else:
            print(f"⚠ {args.log} 에 Public_LB_1-nMAE / Public_LB_FiCR 열이 없다. 건너뜀.")
    for a in (args.add or []):
        rows.append((a[2], float(a[0]), float(a[1])))

    out = report(rows)
    compare(out, args.mine, args.top)
    print(BAR)
    print("읽는 법: 프리미엄이 높다 = 같은 정확도로 밴드를 더 잘 썼다.")
    print("         상대가 총점으로 앞설 때 어느 축에서 앞섰는지를 이걸로 판별한다.")


if __name__ == "__main__":
    main()