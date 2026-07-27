"""
scripts/step23b_pairwise.py
==============================================================================
[도구] step23 의 4개 조건을 모든 쌍으로 짝비교함. 학습 0회, 저장된 OOF 만 읽음.

왜 필요한가
  step23 은 전부 **A 기준**으로만 짝비교했음. 그래서 '격자 산포 효과' 가
  '선택 예산 효과' 와 섞여 있음. 특히 `D vs C`(= K250 에서 산포를 더한 효과)를
  직접 재지 않아 상호작용 −0.0030 이 실재인지 노이즈인지 판정할 수 없음.

  그리고 step23 의 주효과는 **총점 차이**로만 냈으므로 오차막대가 없음.
  여기서는 연도별로 주효과를 계산해 표준편차를 붙임. 그래야 §3.6 규칙 2 로 판정 가능함.

읽는 법
  단순 효과
      B − A   K200 에서 산포를 더한 효과
      D − C   K250 에서 산포를 더한 효과      <- 이 둘이 갈리면 상호작용이 실재함
      C − A   기존 피처만으로 K 를 늘린 효과
      D − B   산포가 있는 상태에서 K 를 늘린 효과
  주효과 (연도별로 계산 후 평균·표준편차)
      격자산포 = ((B−A) + (D−C)) / 2
      선택예산 = ((C−A) + (D−B)) / 2
      상호작용 =  D − B − C + A

판정 (§3.6 규칙 2·6)
  부호 3/3 일치 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수.

실행
    python scripts/step23b_pairwise.py --config configs/config_v13.yaml
==============================================================================
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import CAPACITY_KWH
from src.validation import (quiet_warnings, add_time_keys, total_score, group_score,
                            is_difference_real)

quiet_warnings()
BAR = "=" * 84
NAME = {"A": "A 기준 K200", "B": "B +산포 K200", "C": "C 기준 K250", "D": "D +산포 K250"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", default="./saved_models/_ab_step23")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])
    d = Path(args.dir)

    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()
    yrs = df["_year"].to_numpy()
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]

    O = {}
    for k in "ABCD":
        p = d / f"oof_{k}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음. step23 을 먼저 실행할 것.")
        O[k] = pd.read_csv(p, index_col=0).reindex(index=df.index, columns=targets).astype(float)

    print(BAR); print("조건별 raw OOF")
    for k in "ABCD":
        s = total_score(answer, O[k], targets)
        print(f"  {NAME[k]:14s} raw {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}")

    # 연도별 점수 행렬 (주효과 오차막대용)
    ys = {k: np.array([total_score(answer.loc[yrs == y], O[k].loc[yrs == y], targets)[0]
                       for y in years]) for k in "ABCD"}

    print(BAR); print("단순 효과 — 모든 쌍 짝비교 (§3.6 규칙 2)")
    order = [("B", "A"), ("D", "C"), ("C", "A"), ("D", "B"), ("D", "A"), ("C", "B")]
    for a, b in order:
        print(f"\n  [{NAME[a]}  −  {NAME[b]}]")
        is_difference_real(df, answer, O[a], O[b], targets, name_a=NAME[a], name_b=NAME[b])
        gpos, line = 0, []
        for g in targets:
            cap = CAPACITY_KWH[g]
            av = answer[g].to_numpy(float)
            sb = group_score(av, O[b][g].to_numpy(float), cap)[0]
            sa = group_score(av, O[a][g].to_numpy(float), cap)[0]
            gpos += int(np.isfinite(sa - sb) and sa > sb)
            line.append(f"{g.replace('kpx_','')} {sa-sb:+.4f}")
        print("   그룹별: " + "  ".join(line) + f"   (양수 {gpos}/{len(targets)})")

    print(BAR); print("주효과 — 연도별로 계산해 오차막대를 붙임 (step23 에는 없던 것)")
    eff = {
        "격자 산포": (ys["B"] - ys["A"] + ys["D"] - ys["C"]) / 2,
        "선택 예산": (ys["C"] - ys["A"] + ys["D"] - ys["B"]) / 2,
        "상호작용 ": ys["D"] - ys["B"] - ys["C"] + ys["A"],
    }
    print(f"  {'효과':<10s}{'평균':>10s}{'표준편차':>10s}{'비율':>8s}{'부호일치':>10s}   연도별")
    for k, v in eff.items():
        sd = v.std(ddof=1)
        same = bool(np.all(v > 0) or np.all(v < 0))
        ratio = abs(v.mean()) / sd if sd > 0 else np.inf
        det = "  ".join(f"{y}:{x:+.4f}" for y, x in zip(years, v))
        mark = "  ✅" if (same and ratio > 1) else "  ⚠️"
        print(f"  {k:<10s}{v.mean():+10.4f}{sd:10.4f}{ratio:8.2f}{str(same):>10s}{mark}   [{det}]")

    print(BAR); print("해석 가이드")
    bd, dc = eff["격자 산포"], (ys["D"] - ys["C"])
    ba = ys["B"] - ys["A"]
    print(f"  B−A 평균 {ba.mean():+.4f}   D−C 평균 {dc.mean():+.4f}")
    if np.sign(ba.mean()) == np.sign(dc.mean()) and (np.all(dc > 0) or np.all(dc < 0)):
        print("  → 두 K 값에서 산포 효과의 부호가 같음. **산포 효과는 K 와 독립으로 실재함.**")
    elif not (np.all(dc > 0) or np.all(dc < 0)):
        print("  → D−C 의 연도 부호가 갈림. **상호작용 −0.0030 은 노이즈로 봐야 함.**")
        print("     이 경우 산포 효과는 B−A(부호 3/3, 표준편차 작음)로 판정하는 것이 타당함.")
    else:
        print("  → 두 K 값에서 산포 효과의 부호가 반대. **산포와 선택예산은 대체재임.**")
        print("     K 를 늘릴 것인지 산포를 넣을 것인지 하나만 골라야 함. 둘 다는 손해임.")
    print("\n  ⚠ A~D 는 전부 그룹 독립 구성임(step18 과 동일). v13 은 G1·G3 통합이므로")
    print("    채택 결정 전에 통합을 얹어 재측정해야 함 (§3.5: 한 번에 하나만 바꿈).")


if __name__ == "__main__":
    main()