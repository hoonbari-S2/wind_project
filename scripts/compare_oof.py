"""
scripts/compare_oof.py
==============================================================================
[도구] 저장된 두 OOF 를 연도별 짝비교함. 학습 0회, 수 초.

왜 필요한가
  train.py 가 찍는 `Total` 은 단일 숫자라 노이즈 여부를 알 수 없음.
  연도 표준편차가 0.009 수준인데 버전 간 차이는 0.002 규모이므로,
  **반드시 연도별 짝비교(§3.1)로 봐야 함.** 짝을 지으면 연도 난이도 항이 상쇄되어
  감도가 15배 오름 (±0.018 -> ±0.0012).

판정 (§3.6 규칙 2·6)
  부호 3/3 일치 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수. 넷 다여야 채택.

실행
    python scripts/compare_oof.py --config configs/config_v13.yaml \
        --a ./saved_models/v15 --b ./saved_models/v13
    python scripts/compare_oof.py --config configs/config_v13.yaml \
        --a ./saved_models/v15/oof_preds.csv --b ./saved_models/v13/oof_preds.csv
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
from src.validation import (quiet_warnings, add_time_keys, total_score, group_score,
                            score_by_year, is_difference_real)

quiet_warnings()
BAR = "=" * 80


def load(path, index, targets):
    p = Path(path)
    if p.is_dir():
        p = p / "oof_preds.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, index_col=0).reindex(index=index, columns=targets).astype(float), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--a", required=True, help="새 후보 (디렉토리 또는 csv)")
    ap.add_argument("--b", required=True, help="기준선")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])

    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()

    A, pa = load(args.a, df.index, targets)
    B, pb = load(args.b, df.index, targets)
    na = args.name_a or Path(args.a).name.replace("oof_preds.csv", "") or Path(args.a).parent.name
    nb = args.name_b or Path(args.b).name.replace("oof_preds.csv", "") or Path(args.b).parent.name

    print(BAR)
    print(f"A = {pa}\nB = {pb}")
    for n, o in [(na, A), (nb, B)]:
        s = total_score(answer, o, targets)
        print(f"  {n:10s} raw OOF {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}"
              f"   결측 {int(o.isna().sum().sum()):,}")

    print(BAR); print("연도별 raw")
    score_by_year(df, answer, A, targets, label=na)
    score_by_year(df, answer, B, targets, label=nb)

    print(BAR); print(f"연도짝비교 (§3.6 규칙 2)")
    res = is_difference_real(df, answer, A, B, targets, name_a=na, name_b=nb)

    print(f"\n  [그룹별 — §3.7: 총점으로 합치면 상쇄가 신호를 지움]")
    gpos = 0
    print("    " + f"{'그룹':<12s}{nb:>10s}{na:>10s}{'Δ':>10s}")
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float)
        sb = group_score(a, B[g].to_numpy(float), cap)[0]
        sa = group_score(a, A[g].to_numpy(float), cap)[0]
        gpos += int(np.isfinite(sa - sb) and sa > sb)
        print("    " + f"{g.replace('kpx_',''):<12s}{sb:10.4f}{sa:10.4f}{sa-sb:+10.4f}")

    ok = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2)
    print(f"\n  부호양수 {bool(res and res['mean'] > 0)}  양수그룹 {gpos}/{len(targets)}"
          f"  =>  {'✅ 채택 후보' if ok else '❌ 채택 안 함'}")

    if res and not res["real"]:
        print("\n  해석: 부호가 갈리거나 |평균| ≤ 표준편차임. **두 구성은 구분 불가.**")
        print("        '나쁘다' 가 아니라 '판별 불가' 로 읽고, 더 단순한 쪽을 유지함 (§3.6).")
    elif res and res["real"] and res["mean"] < 0:
        print("\n  해석: 3/3 일관 **악화**. 노이즈가 아니라 실제로 해로움.")
        print("        두 요소가 같은 정보를 담고 있어 함께 쓰면 선택 예산만 잡아먹는 경우가 흔함.")

    print(BAR)


if __name__ == "__main__":
    main()