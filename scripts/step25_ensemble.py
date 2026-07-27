"""
scripts/step25_ensemble.py
==============================================================================
A-3 — 설정 배깅. 개별로는 기준선을 못 넘은 구성들의 단순 평균

근거
  §3.11  배깅(단순 평균)은 이 데이터셋에서 실측으로 이득이 확인된 **유일한** 앙상블 형태임.
         가중 최적화는 폐기(+0.0003±0.0007), 스태킹도 폐기(§3.8).
         그리고 **"다양성은 알고리즘이 아니라 학습 문제에서 뽑는다"** 고 확정함.
  §3.17  시드 배깅은 죽었음 — 시드 간 산포가 1%cap 뿐이라 정산 구간(6%cap)을 못 뒤집음.
         그러나 **설정 간 산포는 훨씬 큼** (v13↔v14 2.4~5.9%cap, step12 측정).

  지금 손에 있는 구성들은 개별로는 전부 v13 을 못 넘었으나
  **서로 다른 피처 집합을 본 같은 통합 구성**임. 평균의 결과는 개별과 다를 수 있음.

후보 (사전 등록 — 결과 보기 전 고정. 전수 탐색 금지)
      E1  v13 + v15                 격자 산포만 다름
      E2  v13 + step24B             안정도만 다름
      E3  v13 + v15 + step24B       셋 다 (같은 통합 구성, 피처만 다름)
      E4  v13 + v12                 통합 유무 (가장 큰 구조 차이)
      E5  v13 + v14                 목적함수 (raw 산포 최대. 단 §3.10 의 정산 구간 잠식 위험)

  ⚠ 다중 비교를 의식해 후보를 5개로 제한함. **통과가 여러 개면 구성 요소가 가장 적은 것**을
     채택함 (동수면 raw 가 높은 것). 이것도 사전 등록임.

판정 (§3.6 규칙 2·6)
  raw OOF 연도짝비교 vs v13. 부호 3/3 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수.

제출 가능성
  E1·E4·E5 는 양쪽 모두 `raw_test_preds.csv` 가 있어 **학습 0회로 즉시 제출 가능**함.
  E2·E3 는 step24 B 의 모델이 저장돼 있지 않으므로 통과 시 재학습이 필요함.
  제출 시에는 **raw 를 먼저 평균하고 후처리를 1회만** 적용함 (§7 v9 이중 후처리 사고).

실행
    python scripts/step25_ensemble.py --config configs/config_v13.yaml
    python scripts/step25_ensemble.py --config configs/config_v13.yaml --make-submission E1
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
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 88

SRC = {
    "v13": "./saved_models/v13/oof_preds.csv",
    "v15": "./saved_models/v15/oof_preds.csv",
    "s24B": "./saved_models/_ab_step24/oof_B.csv",
    "v12": "./saved_models/v12/oof_preds.csv",
    "v14": "./saved_models/v14/oof_preds.csv",
}
TESTSRC = {                      # 제출용 raw test 예측 (있는 것만)
    "v13": "./saved_models/v13/raw_test_preds.csv",
    "v15": "./saved_models/v15/raw_test_preds.csv",
    "v12": "./saved_models/v12/raw_test_preds.csv",
    "v14": "./saved_models/v14/raw_test_preds.csv",
}
CAND = {
    "E1": ["v13", "v15"],
    "E2": ["v13", "s24B"],
    "E3": ["v13", "v15", "s24B"],
    "E4": ["v13", "v12"],
    "E5": ["v13", "v14"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--make-submission", default=None, help="E1 / E4 / E5 중 하나")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])

    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()

    O = {}
    print(BAR); print("구성별 raw OOF")
    for k, p in SRC.items():
        if not Path(p).exists():
            print(f"  {k:5s} ⚠ 없음 ({p}) — 해당 조합은 건너뜀"); continue
        O[k] = pd.read_csv(p, index_col=0).reindex(index=df.index, columns=targets).astype(float)
        s = total_score(answer, O[k], targets)
        print(f"  {k:5s} raw {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}   결측 {int(O[k].isna().sum().sum()):,}")

    # 구성 간 예측 산포 — 배깅이 깎을 대상 (§3.17 과 대조)
    print(BAR); print("v13 대비 예측 산포 (§3.17: 시드 배깅은 1%cap 뿐이라 죽었음. 정산 구간 반폭 6%cap)")
    for k in O:
        if k == "v13":
            continue
        med = [np.nanmedian(np.abs(O[k][g] - O["v13"][g]) / CAPACITY_KWH[g] * 100) for g in targets]
        p90 = [np.nanquantile(np.abs(O[k][g] - O["v13"][g]) / CAPACITY_KWH[g] * 100, 0.9) for g in targets]
        print(f"  v13 vs {k:5s}  중앙값 {np.mean(med):5.2f}%cap   90분위 {np.mean(p90):5.2f}%cap")

    print(BAR); print("사전 등록 후보 판정 (§3.6 규칙 2·6)")
    results = {}
    for name, parts in CAND.items():
        if any(p not in O for p in parts):
            print(f"\n  [{name}] 구성 요소 누락 — 건너뜀"); continue
        E = pd.DataFrame({g: np.clip(np.nanmean([O[p][g].to_numpy(float) for p in parts], axis=0),
                                     0, CAPACITY_KWH[g]) for g in targets}, index=df.index)
        s = total_score(answer, E, targets)
        print(f"\n  [{name} = {' + '.join(parts)}]  raw {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}")
        res = is_difference_real(df, answer, E, O["v13"], targets, name_a=name, name_b="v13")
        gpos, line = 0, []
        for g in targets:
            cap = CAPACITY_KWH[g]; a = answer[g].to_numpy(float)
            s13 = group_score(a, O["v13"][g].to_numpy(float), cap)[0]
            se = group_score(a, E[g].to_numpy(float), cap)[0]
            gpos += int(np.isfinite(se - s13) and se > s13)
            line.append(f"{g.replace('kpx_','')} {se-s13:+.4f}")
        ok = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2)
        print("   그룹별: " + "  ".join(line) + f"   (양수 {gpos}/{len(targets)})")
        sub_ok = all(p in TESTSRC and Path(TESTSRC[p]).exists() for p in parts)
        print(f"   => {'✅ 채택 후보' if ok else '❌ 채택 안 함'}"
              f"   |  즉시 제출 {'가능' if sub_ok else '불가 (재학습 필요)'}")
        results[name] = dict(ok=ok, n=len(parts), raw=s[0], sub=sub_ok, oof=E)

    print(BAR); print("사전 등록 선택 규칙: 통과가 여러 개면 **구성 요소가 가장 적은 것**, 동수면 raw 높은 것")
    passed = [(v["n"], -v["raw"], k) for k, v in results.items() if v["ok"]]
    if not passed:
        print("  통과 후보 없음. 설정 배깅 축 종결 (§3.17 과 함께 기록).")
    else:
        passed.sort()
        win = passed[0][2]
        print(f"  통과: {', '.join(sorted(k for _, _, k in passed))}   =>  **채택: {win}"
              f"** ({' + '.join(CAND[win])}, raw {results[win]['raw']:.4f})")
        if not results[win]["sub"]:
            print("  ⚠ 즉시 제출 불가 — 구성 요소의 raw_test_preds.csv 가 없음. 재학습 후 제출할 것.")
        print(f"\n  제출: python scripts/step25_ensemble.py --config {args.config} --make-submission {win}")

    # ---------------------------------------------------------------- 제출
    if args.make_submission:
        name = args.make_submission
        parts = CAND[name]
        print(BAR); print(f"제출 파일 생성 — {name} = {' + '.join(parts)}")
        missing = [p for p in parts if p not in TESTSRC or not Path(TESTSRC[p]).exists()]
        if missing:
            print(f"  ⛔ raw_test_preds.csv 없음: {missing}. 재학습 후 다시 시도할 것."); return
        base = pd.read_csv(TESTSRC[parts[0]])
        key = "forecast_id" if "forecast_id" in base.columns else "forecast_kst_dtm"
        acc = {g: np.zeros(len(base)) for g in targets}
        for p in parts:
            d = pd.read_csv(TESTSRC[p])
            if not d[key].equals(base[key]):
                d = d.set_index(key).reindex(base[key]).reset_index()
                print(f"  ⚠ {p}: {key} 순서가 달라 정렬 후 결합")
            for g in targets:
                acc[g] += d[g].to_numpy(float) / len(parts)
        sub = base.copy()
        for g in targets:
            sub[g] = np.clip(acc[g], 0, CAPACITY_KWH[g])

        print("  후처리는 **raw 평균 위에 1회만** 적용 (§7 v9 이중 후처리 사고)")
        pp = optimize_postprocessing(answer, results[name]["oof"], mode="piecewise", verbose=False)
        sub = apply_postprocessing(sub, pp)
        for g in targets:
            sub[g] = np.clip(sub[g].to_numpy(float), 0, CAPACITY_KWH[g])
        sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        outdir = Path(cfg["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / (args.out_name or f"submit_ens_{name}.csv")
        cols = [c for c in ["forecast_id", "forecast_kst_dtm"] if c in sub.columns] + targets
        sub[cols].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  🚀 {out}   행 {len(sub):,}  결측 {int(sub[targets].isna().sum().sum())}")
        for g in targets:
            v = sub[g].to_numpy(float)
            print(f"     {g}: 평균 {np.mean(v)/CAPACITY_KWH[g]*100:5.1f}%cap  최대 {np.max(v)/CAPACITY_KWH[g]*100:5.1f}%cap")


if __name__ == "__main__":
    main()