"""
scripts/step31_adversarial.py
==============================================================================
[진행] 연도 간 드리프트 진단 — **학습 데이터 전용** (테스트 파일 0바이트 사용)

──────────────────────────────────────────────────────────────────────────────
규칙 (v2 에서 전면 수정 — 반드시 읽을 것)
──────────────────────────────────────────────────────────────────────────────
유의사항 7: "평가 데이터셋은 제출 파일(예측 결과) 생성을 위한 **추론 목적으로만**
사용할 수 있습니다."
=> 테스트 기상 피처를 판별기 학습·피처 선택 등 추론 외 용도로 쓰는 것은 규칙 위반임.
   v1 설계(train vs 2025 판별)는 이 조항에 걸려 **작성 단계에서 폐기**했음 (실행 전).
   이 스크립트는 ldaps_test / gfs_test 를 **열지 않는다.**

그래서 무엇으로 대신하나: 학습 3개년(2022/23/24) 사이의 드리프트.
  * "해가 바뀔 때 어떤 피처가 흔들리는가" 는 학습 데이터만으로 측정 가능함.
  * 22→23→24 로 **일관되게 표류**하는 피처는 2025 에서도 표류했을 개연성이 높음.
  * 한계를 명시함: 2025 경계에서 일어난 단절(예: NWP 모델 업그레이드)은 원리적으로 못 봄.
    그건 합법적으로는 **리더보드 제출로만** 관측 가능함 (§3.30 (b) 정정).

──────────────────────────────────────────────────────────────────────────────
구성
──────────────────────────────────────────────────────────────────────────────
PART A  연도쌍 판별기 3개 (22vs23 / 22vs24 / 23vs24), 예보블록 GroupKFold AUC.
        AUC 크기 = 해가 바뀔 때 기상 피처 공간이 원래 얼마나 움직이는가.
        (행 단위 분할 금지 — 자기상관으로 AUC 가 부풀어 오름. §3.23 계측 규칙 ②)
PART B  피처별 불안정성 점수 = 세 판별기 gain 몫의 최댓값.
        + 22→23→24 평균의 단조 표류 여부 (일관 방향이면 2025 지속 개연성 ↑).
        + KS 통계 병기. shift_report.csv 저장.
PART C  제외 후보 drop_features.json (상위 N).  제거 후 연도쌍 AUC 재측정 —
        AUC 가 0.5 쪽으로 내려가면 불안정성이 소수 피처에 몰려 있다는 뜻.

이후 (별도 결정): v18 = v13 구성 + drop_features.json 제외 재학습.
  근거는 전부 학습 기간 정보이므로 소명 부담 없음.
  §3.30 규칙상 내부 OOF 는 비열등성 확인용 — 연도 불안정 피처를 뺐을 때 OOF 가
  크게 안 깨지면(소폭 하락 허용) LB 로 판정.

실행
    python scripts/step31_adversarial.py --config configs/config_v13.yaml
    python scripts/step31_adversarial.py --config configs/config_v13.yaml --quick
==============================================================================
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_full_feature_pipeline
from src.validation import quiet_warnings, add_time_keys

quiet_warnings()
BAR = "=" * 92


def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"latitude", "longitude"}
    id_cols = {"forecast_kst_dtm", "grid_id", "data_available_kst_dtm"}
    value_cols = [c for c in df.columns if c not in (id_cols | drop_cols)]
    pv = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=value_cols)
    pv.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in pv.columns]
    pv = pv.reset_index()
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    agg = agg.reset_index()
    avail = (df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first().reset_index()
             if "data_available_kst_dtm" in df.columns else None)
    out = pv.merge(agg, on="forecast_kst_dtm", how="inner")
    if avail is not None:
        out = out.merge(avail, on="forecast_kst_dtm", how="left")
    return out


def cv_auc(X, y, fdays, params, n_splits=5, return_model=False):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    aucs = []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=fdays):
        m = XGBClassifier(**params)
        m.fit(X.iloc[tr], y[tr], verbose=False)
        aucs.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
    model = None
    if return_model:
        model = XGBClassifier(**params)
        model.fit(X, y, verbose=False)
    return float(np.mean(aucs)), float(np.std(aucs)), model


def ks_stat(a, b, bins=200):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100:
        return np.nan
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    g = np.linspace(lo, hi, bins)
    return float(np.abs(np.searchsorted(np.sort(a), g) / len(a)
                        - np.searchsorted(np.sort(b), g) / len(b)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--v13-dir", default="./saved_models/v13")
    ap.add_argument("--out", default="./saved_models/_step31")
    ap.add_argument("--drop-n", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    tr_dir = Path(config["data_paths"]["train_dir"])

    print(BAR)
    print("step31 v2 — 연도 간 드리프트 (학습 데이터 전용. 테스트 파일은 열지 않음)")
    print("🔄 train 피처 조립...")
    ldaps = pd.read_csv(tr_dir / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(tr_dir / "gfs_train.csv", encoding="utf-8-sig")
    w = process_weather_data(ldaps, "ldaps").merge(
        process_weather_data(gfs, "gfs"), on="forecast_kst_dtm",
        how="inner", suffixes=("", "_gfsdup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_gfsdup")])
    TR = add_time_keys(build_full_feature_pipeline(w).replace([np.inf, -np.inf], np.nan))

    feats = joblib.load(Path(args.v13_dir) / "feature_cols.pkl")
    feats = [c for c in feats if c in TR.columns]
    # 달력 피처는 연도 판별에서 정보가 없고(모든 해에 동일 분포) gain 만 흐리므로 유지해도
    # 무해하나, 명시적으로 남겨 둠 — 제외 후보에 뜨면 그 자체가 이상 신호임.
    print(f"   비교 피처 {len(feats)}개 (v13 feature_cols 기준)   행 {len(TR):,}")

    params = dict(n_estimators=100 if args.quick else 200, max_depth=4,
                  learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                  tree_method="hist", eval_metric="auc", random_state=42, n_jobs=-1)
    dv = config.get("model_params", {}).get("device")
    if dv:
        params["device"] = dv
    nsp = 3 if args.quick else 5

    yr = TR["_year"].to_numpy()
    years = [y for y in sorted(pd.unique(yr)) if (yr == y).sum() > 1000]
    pairs = [(years[i], years[j]) for i in range(len(years)) for j in range(i + 1, len(years))]

    # ---------------------------------------------------------------- PART A
    print(BAR)
    print("PART A — 연도쌍 판별 AUC (해가 바뀔 때 기상 피처 공간이 원래 얼마나 움직이나)")
    models, aucs = {}, {}
    for a, b in pairs:
        m = np.isin(yr, [a, b])
        X = TR.loc[m, feats]; y = (yr[m] == b).astype(int)
        auc, sd, mdl = cv_auc(X, y, TR.loc[m, "_fday"].to_numpy(), params, nsp, return_model=True)
        aucs[(a, b)] = auc; models[(a, b)] = mdl
        print(f"   {a} vs {b}:  AUC {auc:.3f} ± {sd:.3f}")
    print(f"   해석: 0.5 = 구분 불가(안정) · 1.0 = 완전 구분(전면 이동).")
    print(f"   인접쌍(22-23, 23-24)보다 원거리쌍(22-24)이 높으면 **누적形 표류**가 있다는 뜻.")

    # ---------------------------------------------------------------- PART B
    print(BAR)
    print(f"PART B — 피처별 불안정성 (세 판별기 gain 몫 최댓값) 상위 {args.drop_n}")
    G = {}
    for k, mdl in models.items():
        imp = np.asarray(mdl.feature_importances_, float)
        G[k] = imp / max(imp.sum(), 1e-12)
    score = np.max(np.vstack([G[k] for k in pairs]), axis=0)
    order = np.argsort(-score)

    rows = []
    print(f"   {'피처':<44}{'불안정':>8}{'KS22-24':>9}{'22평균':>10}{'23평균':>10}{'24평균':>10}{'단조':>6}")
    for k in order[:args.drop_n]:
        c = feats[k]
        m22 = np.nanmean(TR.loc[yr == years[0], c]) if len(years) > 0 else np.nan
        m23 = np.nanmean(TR.loc[yr == years[1], c]) if len(years) > 1 else np.nan
        m24 = np.nanmean(TR.loc[yr == years[2], c]) if len(years) > 2 else np.nan
        mono = "→" if (np.isfinite(m22) and np.isfinite(m23) and np.isfinite(m24)
                       and (m22 < m23 < m24 or m22 > m23 > m24)) else ""
        ksv = ks_stat(TR.loc[yr == years[0], c].to_numpy(float),
                      TR.loc[yr == years[-1], c].to_numpy(float))
        rows.append(dict(feature=c, instability=float(score[k]), ks_22_24=ksv,
                         mean_2022=float(m22), mean_2023=float(m23), mean_2024=float(m24),
                         monotone=bool(mono)))
        print(f"   {c:<44}{score[k]:8.3f}{ksv:9.3f}{m22:10.3f}{m23:10.3f}{m24:10.3f}{mono:>6}")
    print("   '단조 →' = 22→23→24 한 방향 표류. 2025 에도 이어졌을 개연성이 높은 쪽.")

    drops = [r["feature"] for r in rows]
    json.dump(drops, open(outdir / "drop_features.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv(outdir / "shift_report.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- PART C
    print(BAR)
    print(f"PART C — 상위 {args.drop_n}개 제거 후 연도쌍 AUC 재측정")
    keep = [c for c in feats if c not in set(drops)]
    concentrated = 0
    for a, b in pairs:
        m = np.isin(yr, [a, b])
        auc2, sd2, _ = cv_auc(TR.loc[m, keep], (yr[m] == b).astype(int),
                              TR.loc[m, "_fday"].to_numpy(), params, nsp)
        drop_frac = (aucs[(a, b)] - auc2) / max(aucs[(a, b)] - 0.5, 1e-9)
        concentrated += int(drop_frac > 0.5)
        print(f"   {a} vs {b}:  {aucs[(a,b)]:.3f} → {auc2:.3f}   (구분력의 {drop_frac*100:.0f}% 제거)")
    if concentrated >= 2:
        print("\n   => 연도 불안정성이 소수 피처에 집중됨. **v18(제외 재학습) 근거 성립.**")
        print("      train.py 에 --drop-features 패치 후:")
        print("      python main/train.py --config configs/config_v18.yaml --clean-target --top-k 200 \\")
        print("          --joint-groups kpx_group_1,kpx_group_3 --drop-features ./saved_models/_step31/drop_features.json \\")
        print("          --baseline-oof ./saved_models/v13/oof_preds.csv")
    else:
        print("\n   => 불안정성이 넓게 퍼져 있음. 피처 제외로는 부족 — shift_report.csv 를 눈으로 보고")
        print("      계열 단위(예: 특정 NWP 변수군) 원인을 찾을 것.")

    print(BAR)
    print(f"산출물: {outdir}/drop_features.json, shift_report.csv   경과 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()