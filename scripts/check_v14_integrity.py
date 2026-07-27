"""
scripts/check_v14_integrity.py
==============================================================================
saved_models/v14 가 LB 0.629475 를 만든 그 모델이 맞는지 확인한다.

중간에 train.py 를 잘못 덮어써서 재학습이 일어났다면, step10/step11 이 읽은 OOF 가
제출과 다른 모델의 것이 되어 지금까지의 분해가 전부 무효가 된다.
학습 없이 저장된 산출물만 대조한다.

실행:
    python scripts/check_v14_integrity.py
==============================================================================
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECT = {                       # experiment_log.xlsx 의 v14 행에서 온 값
    "raw_oof": 0.6193,
    "nested_post": 0.6298,
    "objective": "reg:absoluteerror",
    "sample_weight": "ficr",
}
BAR = "=" * 74


def main():
    d = Path("./saved_models/v14")
    if not d.exists():
        print(f"❌ {d} 없음"); return
    ok = True

    print(BAR); print("1. validation_report.json — 어떤 조건으로 학습됐나")
    vr = d / "validation_report.json"
    if vr.exists():
        r = json.load(open(vr))
        for k in ["objective", "sample_weight", "low_weight", "clean_target",
                  "top_k", "joint_groups", "raw_oof", "nested_post"]:
            print(f"   {k:15s} {r.get(k)}")
        for k in ["objective", "sample_weight"]:
            if k in r and r[k] != EXPECT[k]:
                print(f"   ❌ {k} 가 {EXPECT[k]} 가 아니다 -> 다른 조건으로 재학습됐다")
                ok = False
        if "sample_weight" not in r:
            print("   ⚠️ sample_weight 키가 없다 -> 패치 이전 train.py 로 돌았을 가능성")
            ok = False
        if "raw_oof" in r and abs(float(r["raw_oof"]) - EXPECT["raw_oof"]) > 0.0006:
            print(f"   ❌ raw_oof {r['raw_oof']:.4f} != 로그값 {EXPECT['raw_oof']}")
            ok = False
    else:
        print("   ⚠️ 없음 (패치 이전 train.py 로 돌았거나 학습이 안 됨)")
        ok = False

    print(BAR); print("2. oof_preds.csv — step10/step11 이 읽은 그 OOF 인가")
    op = d / "oof_preds.csv"
    if op.exists():
        oof = pd.read_csv(op, index_col=0)
        print(f"   행 {len(oof):,}  컬럼 {list(oof.columns)}")
        print(f"   그룹별 평균 예측: "
              + ", ".join(f"{c}={oof[c].mean():,.0f}" for c in oof.columns))
        print("   (step11 이 찍은 B raw OOF 0.6193 과 로그값이 이미 일치했다면 이 파일은 정상)")
    else:
        print("   ❌ 없음"); ok = False

    print(BAR); print("3. 파일 수정 시각 — 재학습 흔적")
    files = ["validation_report.json", "oof_preds.csv", "post_params.pkl",
             "target_scale.json", "raw_test_preds.csv", "feature_cols.pkl"]
    times = {}
    for f in files:
        p = d / f
        if p.exists():
            times[f] = datetime.fromtimestamp(p.stat().st_mtime)
            print(f"   {f:24s} {times[f]:%Y-%m-%d %H:%M:%S}")
    models = sorted(d.glob("model_*.pkl"))
    if models:
        mt = [datetime.fromtimestamp(p.stat().st_mtime) for p in models]
        print(f"   model_*.pkl ({len(models)}개)      "
              f"{min(mt):%Y-%m-%d %H:%M:%S} ~ {max(mt):%Y-%m-%d %H:%M:%S}")
        # 학습 산출물(모델/OOF)이 raw_test_preds 보다 나중이면 = 제출 후 재학습
        if "raw_test_preds.csv" in times and max(mt) > times["raw_test_preds.csv"]:
            print("   ❌ 모델이 raw_test_preds.csv 보다 나중에 만들어졌다 "
                  "-> 제출 이후 재학습됨. LB 0.629475 는 지금 이 모델의 점수가 아니다.")
            ok = False
        else:
            print("   ✅ 모델이 test 예측보다 앞선다 -> 제출한 그 모델이 맞다")

    print(BAR); print("4. raw_test_preds.csv vs submit_v14_no_post.csv")
    rt, sn = d / "raw_test_preds.csv", Path("./submissions/submit_v14_no_post.csv")
    if rt.exists() and sn.exists():
        a, b = pd.read_csv(rt), pd.read_csv(sn)
        cols = [c for c in a.columns if c.startswith("kpx_")]
        if len(a) == len(b):
            diff = max(float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))))
                       for c in cols)
            print(f"   최대 절대차 {diff:.6f} kWh")
            print("   ✅ 동일 (--no-post 는 raw 를 그대로 내보내는 게 맞다)" if diff < 1e-6
                  else "   ❌ 다르다 -> raw_test_preds 가 제출 이후 다시 생성됐다")
            ok = ok and diff < 1e-6
        else:
            print(f"   ❌ 행 수가 다르다 {len(a)} vs {len(b)}"); ok = False
    else:
        print("   ⏭ 둘 중 하나가 없어 건너뜀")

    print(BAR)
    print("✅ 무결성 확인. step10/step11 분해는 유효하다." if ok else
          "❌ 문제 발견. 위 항목을 보고, 필요하면 v14 를 원래 인자로 재학습해야 한다:\n"
          "   python main/train.py --config configs/config_v14.yaml --clean-target --top-k 200 "
          "--joint-groups kpx_group_1,kpx_group_3 --objective mae --sample-weight ficr")


if __name__ == "__main__":
    main()