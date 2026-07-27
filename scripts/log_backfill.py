"""
scripts/log_backfill.py
==============================================================================
[도구] 제출은 했는데 experiment_log.xlsx 에 안 들어간 건을 소급 등록함.

이번 대상 (2026-07-26)
  v13-nopost      LB 0.620418  — §8.1 계측용. 전이 모델 두 번째 관측점
  v15-blend-w50   LB 0.633398  — §3.12 종결용. 정산 초과분 곡선 볼록성 검정

혼합 조건의 Raw OOF 지표는 저장된 OOF 를 직접 섞어 계산함 (하드코딩하지 않음).
`src/logger.py` 의 log_experiment 를 그대로 써서 스키마·복구 로직을 재사용함.

실행
    python scripts/log_backfill.py --config configs/config_v13.yaml
    python scripts/log_backfill.py --config configs/config_v13.yaml --dry-run
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
from src.logger import log_experiment
from src.validation import quiet_warnings, add_time_keys, total_score, score_by_year

quiet_warnings()

# ---- 실측 LB (제출 후 수기 확인값) -------------------------------------------
LB = {
    "v13-nopost":    dict(score=0.6204182985, nmae=0.872184052,  ficr=0.368652545),
    "v15-blend-w50": dict(score=0.6333979843, nmae=0.8650429372, ficr=0.4017530314),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-a", default="./saved_models/v13")
    ap.add_argument("--dir-b", default="./saved_models/v14")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--log", default="experiment_log.xlsx")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])

    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()

    oa = pd.read_csv(Path(args.dir_a) / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=targets).astype(float)
    ob = pd.read_csv(Path(args.dir_b) / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=targets).astype(float)
    w = args.w
    bl = pd.DataFrame({g: np.clip((1 - w) * oa[g].to_numpy(float) + w * ob[g].to_numpy(float),
                                  0, CAPACITY_KWH[g]) for g in targets}, index=df.index)

    s13 = total_score(answer, oa, targets)
    sbl = total_score(answer, bl, targets)
    _, _, sp13 = score_by_year(df, answer, oa, targets, label="v13 raw OOF")
    _, _, spbl = score_by_year(df, answer, bl, targets, label=f"blend w={w} raw OOF")

    rows = [
        dict(key="v13-nopost", version="v13-nopost", post="none",
             raw=s13[0], tot=s13[0], nm=s13[1], fi=s13[2], spread=sp13,
             obj="reg:tweedie", sw="none",
             feat="v13 구성 전부 유지 (정제 타깃 + 폴드별 top200 + G1·G3 통합). 후처리만 제거",
             note=("§8.1 계측용 제출. v13 의 raw LB 를 직접 확보해 전이 모델을 2점으로 만듦. "
                   "여지13 = 0.641458 − 0.620418 = +0.021040. "
                   "사전 예측 두 경로(0.606 / 0.621) 중 '여지 차감' 경로와 0.0006 이내 일치. "
                   "§3.10 대수식 확정: raw14 − raw13 = +0.006445, 검산 오차 0. "
                   "정산구간형 raw 전이율 35% -> 25% 로 정정(step9 OOF +0.0255 -> LB +0.006445). "
                   "raw 정산 초과분 +0.0087 (이론 FICR 0.3600) — v13 최종 +0.0668 중 "
                   "+0.058 을 후처리가 단독 생성함. 후처리는 정확도 −0.0045 를 팔아 FICR +0.0466 을 삼.")),
        dict(key="v15-blend-w50", version="v15-blend-w50", post="piecewise",
             raw=sbl[0], tot=sbl[0], nm=sbl[1], fi=sbl[2], spread=spbl,
             obj=f"blend tweedie×{1-w:.2f} + mae×{w:.2f}", sw="mixed",
             feat=f"v13 raw × {1-w:.2f} + v14 raw × {w:.2f} 결합 후 piecewise 후처리 1회 (학습 0회)",
             note=("§3.12 종결용 제출. 사전 등록 판정 0.6325~0.6385 = '선형' 구간에 착지. "
                   "세 점 실측으로 2차 곡선 확정: total(w) = 0.641458 − 0.020257w + 0.008274w², "
                   "꼭짓점 w=1.224 (구간 밖), 2차항 계수 > 0 이므로 [0,1] 최댓값은 w=0. "
                   "=> 혼합은 어떤 w 에서도 v13 을 이기지 못함. §3.12 [보류] -> [폐기]. "
                   "정산 초과분 +0.0596 (이론 FICR 0.34215) 으로 이미 v14 수준(+0.0600)까지 하락했고 "
                   "정확도도 0.8677 -> 0.8650 로 하락. 중간이 양쪽의 나쁜 점만 취함.")),
    ]

    print("=" * 74)
    for r in rows:
        lb = LB[r["key"]]
        print(f"{r['version']:16s} rawOOF {r['raw']:.4f}  연도표준편차 {r['spread']:.4f}  "
              f"LB {lb['score']:.6f} (1-nMAE {lb['nmae']:.6f} / FICR {lb['ficr']:.6f})")
    if args.dry_run:
        print("\n--dry-run: 기록하지 않음"); return

    for r in rows:
        c = dict(cfg)
        c["version"] = r["version"]
        c["features_summary"] = r["feat"]
        c["notes"] = r["note"]
        log_experiment(config=c, total_score=r["tot"], one_minus_nmae=r["nm"], ficr=r["fi"],
                       execution_time_sec=0.0, validation="loyo", target_kind="clean_cf",
                       es_mode="refit", post_mode=r["post"], objective=r["obj"],
                       sample_weight=r["sw"], raw_oof=r["raw"], year_spread=r["spread"],
                       n_features=812, log_path=args.log)

    # LB 3열은 log_experiment 가 None 으로 두므로 직접 채움
    d = pd.read_excel(args.log)
    for r in rows:
        m = d["Version"] == r["version"]
        if m.any():
            i = d.index[m][-1]
            lb = LB[r["key"]]
            d.loc[i, "Public_LB_Score"] = lb["score"]
            d.loc[i, "Public_LB_1-nMAE"] = lb["nmae"]
            d.loc[i, "Public_LB_FiCR"] = lb["ficr"]
    d.to_excel(args.log, index=False, engine="openpyxl")
    print(f"\n✅ {args.log} 에 2행 등록 + LB 3열 기입 완료  (총 {len(d)}행)")
    print(d[["Version", "Raw OOF", "Val_Total Score", "Public_LB_Score",
             "Public_LB_1-nMAE", "Public_LB_FiCR"]].tail(6).to_string(index=False))


if __name__ == "__main__":
    main()