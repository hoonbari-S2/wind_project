"""
scripts/step22_blend_submit.py
==============================================================================
§3.12 종결용 — Tweedie(v13) × MAE+W(v14) 혼합 제출 파일 생성

학습 0회. 저장된 raw 예측만 섞고 후처리를 **딱 한 번** 적용함.

왜 이 제출인가 (v13-nopost LB 확보 후 갱신된 사전 등록)
  끝점 두 개가 이제 실측으로 확정됨.
      w=0 (v13):  raw 0.620418  여지 +0.021040  총점 0.641458
      w=1 (v14):  raw 0.626863  여지 +0.002612  총점 0.629475
  raw 는 w 에 대해 단조 증가(§3.12), 여지는 단조 감소.
  둘 다 선형이면 total(0.5) = 0.6355 이고 w=0 를 못 넘음.
  => **이 제출은 수준이 아니라 여지(w) 의 볼록성을 검정함.**

사전 등록 판정 (결과 보기 전 고정)
      > 0.6415          여지가 강하게 볼록. 내부 최적점 실재 -> w=0.35 1회 추가
      0.6325 ~ 0.6385   선형. §3.12 혼합 트랙 종결
      < 0.6325          오목. 어느 w 에서도 짐. §3.12 종결 + 'raw 단조증가' 해석 재검토

주의 — 이중 후처리 금지 (§7 v9 사고)
  반드시 **raw 예측을 먼저 섞고** 그 위에 후처리를 1회만 적용함.
  각자 후처리한 것을 섞으면 안 됨.

후처리 파라미터는 가진 데이터 전부(3년)로 적합함. 배포 절차와 동일하게 맞춤.

실행
    python scripts/step22_blend_submit.py --config configs/config_v13.yaml --w 0.5
    python scripts/step22_blend_submit.py --config configs/config_v13.yaml --w 0.35
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
from src.validation import quiet_warnings, add_time_keys, total_score, group_score
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 78


def _cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def implied_ficr(nmae):
    """그 NMAE 가 자연히 수반하는 이론 FICR (§2.0)."""
    s = float(nmae) * np.sqrt(np.pi / 2)
    p6 = 2 * _cdf(0.06 / s) - 1
    p8 = 2 * _cdf(0.08 / s) - 1
    return float(p6 + 0.75 * (p8 - p6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-a", default="./saved_models/v13", help="w=0 쪽 (Tweedie)")
    ap.add_argument("--dir-b", default="./saved_models/v14", help="w=1 쪽 (MAE+W)")
    ap.add_argument("--w", type=float, default=0.5, help="v14 가중치")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    da, dbb = Path(args.dir_a), Path(args.dir_b)
    w = float(args.w)
    out_name = args.out_name or f"submit_v15_blend_w{int(round(w*100)):02d}.csv"

    # ------------------------------------------------------------ 라벨 + OOF
    tdir = Path(cfg["data_paths"]["train_dir"])
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    df = add_time_keys(labels)
    answer = df[targets].copy()

    oa = pd.read_csv(da / "oof_preds.csv", index_col=0).reindex(index=df.index, columns=targets).astype(float)
    ob = pd.read_csv(dbb / "oof_preds.csv", index_col=0).reindex(index=df.index, columns=targets).astype(float)

    print(BAR); print(f"혼합 w = {w:.2f}   ( {da.name} × {1-w:.2f}  +  {dbb.name} × {w:.2f} )")
    for nm, o in [(da.name, oa), (dbb.name, ob)]:
        s = total_score(answer, o, targets)
        print(f"  {nm} raw OOF {s[0]:.4f}  (1-NMAE {s[1]:.4f} / FICR {s[2]:.4f})")

    blend_oof = pd.DataFrame({g: (1 - w) * oa[g].to_numpy(float) + w * ob[g].to_numpy(float)
                              for g in targets}, index=df.index)
    for g in targets:
        blend_oof[g] = np.clip(blend_oof[g], 0, CAPACITY_KWH[g])
    sblend = total_score(answer, blend_oof, targets)
    print(f"  혼합    raw OOF {sblend[0]:.4f}  (1-NMAE {sblend[1]:.4f} / FICR {sblend[2]:.4f})")
    print(f"    정산 초과분 {sblend[2] - implied_ficr(1 - sblend[1]):+.4f}  "
          f"(이론 FICR {implied_ficr(1 - sblend[1]):.4f})")

    # ------------------------------------------------------------ test 예측
    print(BAR); print("test raw 예측 결합 — **반드시 raw 를 먼저 섞고 후처리는 1회만** (§7 v9 사고)")
    pa = pd.read_csv(da / "raw_test_preds.csv")
    pb = pd.read_csv(dbb / "raw_test_preds.csv")
    if len(pa) != len(pb):
        raise ValueError(f"행 수 불일치: {len(pa)} vs {len(pb)}")
    key = "forecast_id" if ("forecast_id" in pa.columns and "forecast_id" in pb.columns) else "forecast_kst_dtm"
    if not pa[key].equals(pb[key]):
        print(f"  ⚠ {key} 순서가 달라 정렬 후 결합함")
        pb = pb.set_index(key).reindex(pa[key]).reset_index()
    print(f"  정렬 키 {key}   행 {len(pa):,}")

    sub = pa.copy()
    for g in targets:
        v = (1 - w) * pa[g].to_numpy(float) + w * pb[g].to_numpy(float)
        sub[g] = np.clip(v, 0, CAPACITY_KWH[g])
        d = np.abs(pa[g].to_numpy(float) - pb[g].to_numpy(float)) / CAPACITY_KWH[g] * 100
        print(f"  {g}: 두 모델 예측 차이 중앙값 {np.median(d):.2f}%cap  90분위 {np.quantile(d,0.9):.2f}%cap")

    # ------------------------------------------------------------ 후처리 1회
    print(BAR); print("후처리 적합 (혼합 OOF 전체 3년) 후 test 에 1회 적용")
    pp = optimize_postprocessing(answer, blend_oof, mode="piecewise", verbose=False)
    for g in targets:
        print(f"  {g}: out_knots [" + " ".join(f"{v:.3f}" for v in pp[g]["out_knots"]) + "]"
              f"  zero_th {pp[g]['zero_th']/CAPACITY_KWH[g]:.3f}cap")
    post_oof = apply_postprocessing(blend_oof.copy(), pp)
    spost = total_score(answer, post_oof, targets)
    print(f"\n  [참고] 혼합 OOF 후처리 후 {spost[0]:.4f}  (in-sample 적합이라 과대추정. 판정에 쓰지 않음 §3.6)")

    sub_post = apply_postprocessing(sub.copy(), pp)
    for g in targets:
        sub_post[g] = np.clip(sub_post[g].to_numpy(float), 0, CAPACITY_KWH[g])

    sub_post["forecast_kst_dtm"] = pd.to_datetime(sub_post["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    outdir = Path(cfg["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / out_name
    cols = [c for c in ["forecast_id", "forecast_kst_dtm"] if c in sub_post.columns] + targets
    sub_post[cols].to_csv(out, index=False, encoding="utf-8-sig")

    print(BAR)
    print(f"🚀 제출 파일: {out}")
    print(f"   행 {len(sub_post):,}   결측 {int(sub_post[targets].isna().sum().sum())}개")
    for g in targets:
        v = sub_post[g].to_numpy(float)
        print(f"   {g}: 평균 {np.mean(v)/CAPACITY_KWH[g]*100:5.1f}%cap  "
              f"최대 {np.max(v)/CAPACITY_KWH[g]*100:5.1f}%cap  0인 행 {int((v==0).sum()):,}")
    print(BAR)
    print("사전 등록 판정 (제출 후 LB 와 대조)")
    print("   > 0.6415        여지 강하게 볼록 -> w=0.35 1회 추가")
    print("   0.6325~0.6385   선형 -> §3.12 혼합 트랙 종결")
    print("   < 0.6325        오목 -> §3.12 종결 + 'raw 단조증가' 해석 재검토")
    print("   (선형 가정 예측치 0.6355)")


if __name__ == "__main__":
    main()