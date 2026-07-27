"""
scripts/step5b_hybrid_eval.py
==============================================================================
step5 가 저장한 OOF 를 조합만 바꿔 재평가한다. **재학습 없음 — 수 초면 끝난다.**

step5 결과 (raw, 그룹별):
    조건        G1       G2       G3
    A 독립    0.5962   0.6064   0.5578
    B 통합    0.6009   0.6011   0.5762    ← G3 +0.0184, G2 −0.0053

borrowed strength 의 표준 패턴이다. 데이터가 적은 쪽(G3, 라벨 17,538)이 이득,
많은 쪽(G2, 26,201)이 손해. G3 에만 통합을 쓰면 손실 없이 이득만 가져간다.

**사후 선택이 아니다.** G3 만 라벨이 33% 적다는 것은 실험 전에 아는 사실이고,
적은 쪽이 borrowing 에서 이득을 본다는 것도 원칙적으로 예측 가능한 방향이다.

또 하나: 2022년은 G3 라벨이 없어 G1·G2 만으로 채점된다. 통합의 이득은 전부
G3 에서 오므로 2022년에는 G2 손해만 남는다. 테스트(2025)는 세 그룹이 모두 있으므로
**2022년은 테스트 상황을 대표하지 않는다.** 판정에서 이를 분리해 표기한다.

실행:
    python scripts/step5b_hybrid_eval.py --config configs/config_v12.yaml
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
                            score_by_year, is_difference_real, nested_postprocess_score)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ab-dir", default="./saved_models/_ab_step5")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])
    ad = Path(args.ab_dir)

    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()

    oof = {}
    for tag, f in [("A 독립", "oof_A.csv"), ("B 통합", "oof_B.csv"), ("C 통합+균형", "oof_C.csv")]:
        p = ad / f
        if p.exists():
            oof[tag] = pd.read_csv(p, index_col=0).reindex(index=df.index, columns=targets)
    if "A 독립" not in oof or "B 통합" not in oof:
        print(f"❌ {ad}/oof_A.csv, oof_B.csv 가 필요하다. step5 를 먼저 실행할 것.")
        return

    # ---- 조합 생성 ----
    A, B = oof["A 독립"], oof["B 통합"]
    hyb = A.copy(); hyb["kpx_group_3"] = B["kpx_group_3"]
    oof["D G3만통합"] = hyb
    blend = A.copy()
    for g in targets:
        blend[g] = 0.5 * A[g] + 0.5 * B[g]
    oof["E 5:5 블렌딩"] = blend
    hb = A.copy(); hb["kpx_group_3"] = 0.5 * A["kpx_group_3"] + 0.5 * B["kpx_group_3"]
    oof["F G3만 블렌딩"] = hb
    # 그룹별 짝비교가 지지하는 조합: G1 통합(3/3 +), G2 독립(3/3 −), G3 통합(2/2 +)
    g_opt = A.copy()
    g_opt["kpx_group_1"] = B["kpx_group_1"]
    g_opt["kpx_group_3"] = B["kpx_group_3"]
    oof["G 그룹별최적"] = g_opt
    # 보수판: G1·G3 는 블렌딩
    g_bl = A.copy()
    for g in ("kpx_group_1", "kpx_group_3"):
        g_bl[g] = 0.5 * A[g] + 0.5 * B[g]
    oof["H 그룹별블렌딩"] = g_bl

    print(BAR); print("그룹별 점수 (raw OOF)")
    print(f"  {'조건':16s}" + "".join(f"{g.replace('kpx_','') :>13s}" for g in targets) + f"{'평균':>11s}")
    for n, o in oof.items():
        ss = [group_score(answer[g].to_numpy(float), o[g].to_numpy(float), CAPACITY_KWH[g])[0]
              for g in targets]
        print(f"  {n:16s}" + "".join(f"{v:13.4f}" for v in ss) + f"{np.mean(ss):11.4f}")

    # ---- 그룹별 연도별 짝비교 (통합 vs 독립) ----
    print(BAR); print("그룹별·연도별 짝비교 — 통합(B) − 독립(A)")
    yrs = df["_year"].to_numpy()
    print(f"  {'그룹':14s}" + "".join(f"{y:>11d}" for y in sorted(pd.unique(yrs))) + f"{'평균':>10s}")
    for g in targets:
        row, vals = "", []
        for y in sorted(pd.unique(yrs)):
            m = (yrs == y) & answer[g].notna().to_numpy()
            if m.sum() < 200:
                row += f"{'라벨없음':>11s}"; continue
            sa = group_score(answer.loc[m, g].to_numpy(float), A.loc[m, g].to_numpy(float), CAPACITY_KWH[g])[0]
            sb = group_score(answer.loc[m, g].to_numpy(float), B.loc[m, g].to_numpy(float), CAPACITY_KWH[g])[0]
            vals.append(sb - sa); row += f"{sb-sa:+11.4f}"
        print(f"  {g:14s}{row}{np.mean(vals):+10.4f}")
    print("\n  -> G3 는 2022년 라벨이 없다. 그 해 총점은 G1·G2 만으로 계산되므로")
    print("     통합의 이득(전부 G3 에서 옴)이 반영되지 않고 G2 손해만 남는다.")
    print("     그룹별로는 부호가 완벽히 일관되지만 총점으로 합치면 상쇄된다.")
    print("     -> 그룹별 판정이 총점 판정보다 강력하다. 조합 G/H 가 이 결과를 반영한 것.")

    # ---- 전체 판정 (2022 포함 / 제외) ----
    print(BAR); print("전체 판정")
    for n, o in oof.items():
        score_by_year(df, answer, o, targets, label=n)
    print()
    for n in [k for k in oof if k != "A 독립"]:
        print(f"  --- {n} vs A 독립 (전체 3개 연도) ---")
        is_difference_real(df, answer, oof[n], oof["A 독립"], targets, name_a=n, name_b="A 독립")
    print("\n  --- 2022 제외 (G3 라벨이 있는 해만 = 테스트 상황) ---")
    d23 = df[df["_year"] != 2022]
    for n in [k for k in oof if k != "A 독립"]:
        print(f"  {n}:")
        is_difference_real(d23, answer.loc[d23.index], oof[n].loc[d23.index],
                           oof["A 독립"].loc[d23.index], targets, name_a=n, name_b="A 독립")

    # ---- 후처리 후 ----
    best = max(oof, key=lambda t: total_score(answer, oof[t], targets)[0])
    print(BAR); print(f"후처리 후 (A 독립 vs 최고 {best})")
    post = {}
    for n in dict.fromkeys(["A 독립", best]):
        print(f"  [{n}]")
        post[n], _, _ = nested_postprocess_score(df, answer, oof[n], targets,
                                                 optimize_postprocessing, apply_postprocessing,
                                                 mode="piecewise", verbose=True)
    if len(post) == 2:
        print()
        is_difference_real(df, answer, post[best], post["A 독립"], targets,
                           name_a=best + "+후처리", name_b="A+후처리")
        print("  (2022 제외)")
        is_difference_real(d23, answer.loc[d23.index], post[best].loc[d23.index],
                           post["A 독립"].loc[d23.index], targets,
                           name_a=best + "+후처리", name_b="A+후처리")
    print(BAR)


if __name__ == "__main__":
    main()