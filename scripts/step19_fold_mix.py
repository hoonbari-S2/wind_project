"""
scripts/step19_fold_mix.py
==============================================================================
[구조] inference.py 가 3년 학습 모델을 1/4 가중으로 섞고 있는 문제

발견 경위
  train_labels.csv 범위가 2022-01-01 01:00 ~ **2025-01-01 00:00** 이라
  마지막 행 하나의 `_year` 가 2025 임. `make_folds` 는 `len(va)==0` 만 거르므로
  검증행이 1개뿐인 **val2025 폴드가 생성됨**. `run_cv` 도 `len(tr_fit)<100` 만 보므로 통과함.

      val2022 -> 2023+2024 학습 (약 17,500행)
      val2023 -> 2022+2024 학습
      val2024 -> 2022+2023 학습
      val2025 -> 2022+2023+2024 학습  (26,303행)  <- 사실상 3년 전체 모델

  `saved_models/v13/model_kpx_group_1_val2025.pkl` 와 `model_joint_val2025.pkl` 이 실제로 존재함.
  그리고 inference.py 는 `model_{target}_*.pkl` 을 glob 해서 **4개를 전부 1/4 가중으로 평균**함.
  => 제출 예측의 25% 가 3년 전체 학습 모델, 75% 가 2년 학습 모델임.

왜 문제인가 (누설은 없음 — 2025 라벨 자체가 없음)
  1. **§3.5 의 전이율 논증이 쓰인 대로 성립하지 않음.** "LOYO 는 2년으로 학습하지만 최종 모델은
     3년으로 학습하므로 정규화형 이득은 배포하면 줄어든다" 가 정규화형 전이율 26% 의 근거인데,
     실제 배포는 3년 모델이 아니라 **3:1 혼합**임.
  2. **§3.11 의 폴드 평균 실험(step10)에 3년 모델이 포함되지 않았음.** 그 표는 실제 배포 구성을
     설명하지 않음.
  3. 무엇보다 **의도한 설계가 아님.** 우연히 그렇게 된 것이며, 지금은 선택된 적이 없음.

왜 OOF 로 못 재는가
  val2025 모델은 학습 데이터를 전부 보았으므로 out-of-fold 예측이 존재하지 않음.
  => **LB 가 유일한 계측기임.** §8.7 개정판의 '구조 축은 LB 로 판정' 에 해당함.

만드는 것 (학습 0회, 저장된 모델로 추론만)
      A  all      : val2022~2025 전부 평균          (현행)
      B  full     : val2025 단독                    (3년 전체 학습)
      C  loyo     : val2022~2024 평균               (순수 2년 모델 평균)

판정 (사전 등록 — 결과 보기 전 고정)
  0단계. 세 변형의 test 예측 차이 **중앙값이 1%cap 미만이면 제출하지 않음.**
         정산 구간 반폭이 6%cap 이므로 그보다 한참 작으면 순위를 못 뒤집음.
  1단계. 1%cap 이상이면 **B(full) 1회 제출.** 기준은 현행 A = 0.641458.
         > 0.6435          3년 학습이 유의하게 유리 -> v15 기본 구성을 3년 전체 학습으로 변경
         0.6395 ~ 0.6435   구분 불가 -> 현행 유지. 단 §3.5 논증은 '3:1 혼합' 으로 다시 씀
         < 0.6395          폴드 평균이 유리 -> §3.11 이 배포 구성에서도 성립. 현행 확정
  2단계. B 가 0.6435 를 넘으면 C 도 1회 제출해 '평균 효과' 와 '학습량 효과' 를 분리함.
         (B > A > C 면 학습량 효과 / B ≈ C > A 면 평균이 오히려 해로운 것)

실행
    python scripts/step19_fold_mix.py --config configs/config_v13.yaml
    python scripts/step19_fold_mix.py --config configs/config_v13.yaml --variants full,loyo
==============================================================================
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_full_feature_pipeline
from src.postprocessing import apply_postprocessing
from src.validation import GROUP_SPEC, CREF, quiet_warnings

quiet_warnings()
BAR = "=" * 78
VARIANTS = {"all": None, "full": ["val2025"], "loyo": ["val2022", "val2023", "val2024"]}


def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    idc = {"forecast_kst_dtm", "grid_id", "data_available_kst_dtm"}
    vc = [c for c in df.columns if c not in (idc | {"latitude", "longitude"})]
    piv = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=vc)
    piv.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in piv.columns]
    agg = df.groupby("forecast_kst_dtm")[vc].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    out = piv.reset_index().merge(agg.reset_index(), on="forecast_kst_dtm", how="inner")
    if "data_available_kst_dtm" in df.columns:
        av = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first().reset_index()
        out = out.merge(av, on="forecast_kst_dtm", how="left")
    return out


def joint_X(base_X, target):
    sp = GROUP_SPEC[target]
    d = base_X.astype(np.float32).copy()
    for k in range(3):
        d[f"grp_{k}"] = np.float32(1.0 if sp["gid"] == k else 0.0)
    d["grp_is_vestas"] = np.float32(sp["is_vestas"])
    d["grp_n_turb"] = np.float32(sp["n_turb"])
    d["grp_rotor_d"] = np.float32(sp["rotor_d"])
    return d


def predict(md, X, targets, caps, joint_groups, scale, folds=None):
    out = {}
    for t in targets:
        cap = caps[t]
        if t in joint_groups:
            paths = sorted(md.glob("model_joint_*.pkl"))
            Xt = joint_X(X, t)
        else:
            paths = sorted(md.glob(f"model_{t}_*.pkl"))
            Xt = X
        if folds:
            paths = [p for p in paths if p.stem.split("_")[-1] in folds]
        if not paths:
            raise FileNotFoundError(f"{t}: 조건에 맞는 모델 없음 (folds={folds})")
        ub = 1.15 * (CREF if t in joint_groups else cap) if scale else cap
        pr = np.zeros(len(X))
        for p in paths:
            fold = p.stem.split("_")[-1]
            fc = md / (f"featcols_joint_{fold}.pkl" if t in joint_groups
                       else f"featcols_{t}_{fold}.pkl")
            Xi = Xt[joblib.load(fc)] if fc.exists() else Xt
            pr += np.clip(joblib.load(p).predict(Xi), 0, ub) / len(paths)
        if scale:
            pr = pr * scale[t]
        out[t] = np.clip(pr, 0, cap)
        print(f"    {t}: 모델 {len(paths)}개 ({', '.join(p.stem.split('_')[-1] for p in paths)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", default="all,full,loyo")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    caps = cfg["capacity_kwh"]
    md = Path(cfg["data_paths"]["save_model_dir"])
    td = Path(cfg["data_paths"]["test_dir"])
    sd = Path(cfg["data_paths"]["submission_dir"]); sd.mkdir(parents=True, exist_ok=True)

    print(BAR); print("저장된 폴드 모델 확인")
    for t in targets:
        fs = [p.stem.split("_")[-1] for p in sorted(md.glob(f"model_{t}_*.pkl"))]
        print(f"  {t}: {fs}")
    jf = sorted(p.stem.split("_")[-1] for p in md.glob("model_joint_*.pkl"))
    print(f"  joint: {jf}")
    if "val2025" not in jf and not any("val2025" in str(p) for p in md.glob("model_*val2025*.pkl")):
        print("  ⚠ val2025 모델이 없음 — 이 실험의 전제가 성립하지 않음. 중단 권장.")

    print(BAR); print("test 피처 생성")
    cands = [Path("./sample_submission.csv"), td / "sample_submission.csv", Path("./data/sample_submission.csv")]
    sp = next((c for c in cands if c.exists()), None) or next(iter(Path(".").rglob("sample_submission.csv")))
    sample = pd.read_csv(sp, encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    w = process_weather_data(pd.read_csv(td / "ldaps_test.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(td / "gfs_test.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    feature_cols = joblib.load(md / "feature_cols.pkl")
    X = build_full_feature_pipeline(
        sample[["forecast_id", "forecast_kst_dtm"]].merge(w, on="forecast_kst_dtm", how="left")
    ).replace([np.inf, -np.inf], np.nan).reindex(columns=feature_cols)
    jg = json.load(open(md / "joint_groups.json")) if (md / "joint_groups.json").exists() else []
    scale = json.load(open(md / "target_scale.json")) if (md / "target_scale.json").exists() else None
    pp = joblib.load(md / "post_params.pkl")
    print(f"  행 {len(X):,}  피처 {len(feature_cols)}  통합그룹 {jg}")

    print(BAR)
    raws = {}
    for v in [x.strip() for x in args.variants.split(",")]:
        print(f"[{v}]  folds = {VARIANTS[v] or '전부'}")
        raws[v] = predict(md, X, targets, caps, jg, scale, VARIANTS[v])

    print(BAR); print("변형 간 예측 차이 (0단계 판정: 중앙값 1%cap 미만이면 제출하지 않음)")
    keys = list(raws)
    gate = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            for t in targets:
                d = np.abs(raws[a][t] - raws[b][t]) / caps[t] * 100
                gate = max(gate, float(np.median(d)))
                print(f"  {a:5s} vs {b:5s}  {t}: 중앙값 {np.median(d):5.2f}%cap  "
                      f"90분위 {np.quantile(d,0.9):5.2f}%cap  최대 {np.max(d):5.2f}%cap")
    print(f"\n  최대 중앙값 {gate:.2f}%cap  =>  "
          + ("✅ 1%cap 이상. 제출 가치 있음" if gate >= 1.0
             else "❌ 1%cap 미만. 정산 구간(6%cap)을 못 뒤집음. 제출하지 않음"))

    print(BAR)
    for v, r in raws.items():
        sub = sample[["forecast_id", "forecast_kst_dtm"]].copy()
        for t in targets:
            sub[t] = r[t]
        sub.to_csv(md / f"raw_test_preds_{v}.csv", index=False)
        sub = apply_postprocessing(sub, pp)
        sub["forecast_kst_dtm"] = sub["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out = sd / f"submit_v13_fold_{v}.csv"
        sub.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  {v:5s} -> {out}   " + "  ".join(
            f"{t.replace('kpx_group_','G')} 평균 {np.mean(sub[t])/caps[t]*100:.1f}%cap" for t in targets))

    print(BAR); print("사전 등록 판정 (기준: 현행 A = 0.641458)")
    print("  B(full) 제출 결과")
    print("    > 0.6435          3년 학습이 유의하게 유리 -> v15 기본 구성 변경")
    print("    0.6395 ~ 0.6435   구분 불가 -> 현행 유지. §3.5 논증만 '3:1 혼합' 으로 재작성")
    print("    < 0.6395          폴드 평균이 유리 -> §3.11 이 배포 구성에서도 성립. 현행 확정")
    print("  B > 0.6435 인 경우에만 C(loyo) 추가 제출로 '평균 효과 vs 학습량 효과' 분리")


if __name__ == "__main__":
    main()