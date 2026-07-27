"""
scripts/step30_quantile_utility.py
==============================================================================
B-2b — 분위회귀 + 행별 기대효용 배치.  §3.24 측정이 지목한 '배치 절반' 의 구현

──────────────────────────────────────────────────────────────────────────────
왜 지금 이것인가 (§3.24 · §3.25, 2026-07-27 측정)
──────────────────────────────────────────────────────────────────────────────
  * 1위는 우리 κ-곡선 위에 있음(프리미엄 ≈0) — 그러나 필드 4팀(#6/#19/#28/#40)은
    곡선보다 +0.013~0.031 위에 떠 있음. **같은 정확도에서 그만큼의 FICR 이 실현 가능함.**
  * 우리 단조 후처리는 그 함수 공간의 국소최적(§3.15)이고, 조건부 '이동' 도 닫혔음
    (step28 PART 5: 근접실패 과대비중 52~65% 로 대칭). 남은 통로는 **행별 분포 기반 배치**뿐.
  * 산식의 비대칭이 이 축의 이론적 근거임: 평균 발전량 행 하나를 밴드에 넣는 가치는
    NMAE 페널티 환산으로 **이동거리 최대 ~100%cap 어치** (C·a/Ā). 점예측을 어디에 두느냐가
    L1 최적(중앙값)과 크게 다를 수 있는 구조임. 단 그 이득은 '밴드에 들어갈 확률' 로
    희석되므로, 행별 조건부 분포를 알아야만 답이 나옴. -> 분위회귀.

  §3.8 따름정리에 안 걸림 — NWP 유도량 재계산이 아니라 **학습 문제(손실)의 변경**임.
  step17(B-2) 기각과 다름 — 저건 조건부 기댓값 '한 개' 였고 이건 분포 전체임.

──────────────────────────────────────────────────────────────────────────────
구성 — v13 과 조건을 정확히 맞춤 (차이는 손실뿐)
──────────────────────────────────────────────────────────────────────────────
  * 데이터 조립·정제 타깃·scale: main/train.py 와 동일 로직.
  * **피처: v13 이 저장한 폴드별 featcols pkl 을 그대로 재사용** (top-k 재선택 없음
    -> 선택 분산 제거, §3.5 단일 변경 원칙).
  * 통합 구성 미러링: G1·G3 = long-format 통합 모델, G2 = 독립 (v13 과 동일).
  * 모델: XGBoost objective="reg:quantileerror", quantile_alpha 5개 동시 학습.
    (xgboost >= 2.0. GPU 에서 quantileerror 가 에러나면 --device cpu 로 재실행)
  * ES: 블록 단위 내부 15% 분할 + early_stopping_rounds (fit_no_leak 의 'inner' 모드 상당).

──────────────────────────────────────────────────────────────────────────────
배치(placement) — 후처리를 대체함
──────────────────────────────────────────────────────────────────────────────
  행별 조건부 분포 p(a|x) 를 분위 5점의 조각선형 역CDF 로 근사하고,
  후보 f 격자에서 기대효용을 직접 최대화함.

      U(f) = E_a[ 1{a >= 0.10C} · ( −|f−a| + λ · a · price(|f−a|/C) ) ]
      λ = C / (4·Ā),  Ā = 학습기간 채점행 평균 발전량 (그룹별 상수)

  이 λ 가 산식의 두 반쪽(NMAE 항, FICR 항)의 정확한 상대가중임 (상수항 소거 유도).
  채점 필터 a >= 0.10C 가 적분 안에 들어 있어 '채점 안 될 행' 은 자동으로 자유로워짐.
  **적합 파라미터가 없음** — λ·Ā 는 학습 라벨의 기술통계이고 절점·배율 따위를 안 맞춤.
  따라서 §3.6 이 문제 삼은 '2년 적합 -> 1년 적용' 검정력 문제가 구조적으로 없음.

──────────────────────────────────────────────────────────────────────────────
사전 등록 판정 (결과 보기 전 고정)
──────────────────────────────────────────────────────────────────────────────
  게이트 1 (정확도 비열등성) — 분위 0.5(중앙값) OOF vs v13 raw OOF 연도짝비교.
      '3/3 일관 악화 ∧ |평균|>표준편차' 면 **중단** (분위 학습이 정확도를 깎는 것).
      v14 전례 경고: MAE 계열이 raw 를 올리고 LB 총점을 깎은 적이 있음(§3.10).
  게이트 2 (배치 이득, 참고) — placed OOF vs v13 in-sample 후처리 OOF 연도짝비교.
      부호 3/3 ∧ 양수 ∧ |평균|>표준편차 ∧ 그룹 2개 이상 양수.
      3원 분해도 같이 봄: [v13후처리] vs [분위중앙값+후처리] vs [placed]
      -> 이득이 '분위 학습' 에서 오는지 '배치' 에서 오는지 분리.
  최종 판정은 LB (§3.6 규칙 4). 제출 파일은 게이트 2 통과 시에만 만들 것.

실행
    python scripts/step30_quantile_utility.py --config configs/config_v13.yaml
    # 빠른 예비 (트리 300)
    python scripts/step30_quantile_utility.py --config configs/config_v13.yaml --quick
    # 학습 재사용 (배치·평가만 다시)
    python scripts/step30_quantile_utility.py --config configs/config_v13.yaml --skip-train
    # 제출 파일까지
    python scripts/step30_quantile_utility.py --config configs/config_v13.yaml --make-submission
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
from src.utils import seed_everything, CAPACITY_KWH
from src.features import build_full_feature_pipeline
from src.validation import (quiet_warnings, add_time_keys, make_folds, inner_es_split,
                            total_score, group_score, score_by_year, is_difference_real,
                            build_long, GROUP_SPEC, CREF)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 92
ALPHAS = np.array([0.05, 0.25, 0.50, 0.75, 0.95])


# ------------------------------------------------------------------ 데이터 (train.py 미러)
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


def assemble(config, scada_dir="./data/scada_derived"):
    """train.py 의 데이터 경로를 그대로 밟음 (clean-target 포함)."""
    tr = Path(config["data_paths"]["train_dir"])
    labels = pd.read_csv(tr / "train_labels.csv", encoding="utf-8-sig")
    ldaps = pd.read_csv(tr / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(tr / "gfs_train.csv", encoding="utf-8-sig")
    w = process_weather_data(ldaps, "ldaps").merge(
        process_weather_data(gfs, "gfs"), on="forecast_kst_dtm",
        how="inner", suffixes=("", "_gfsdup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_gfsdup")])
    base = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    base["forecast_kst_dtm"] = pd.to_datetime(base["forecast_kst_dtm"])
    df = base.merge(w, on="forecast_kst_dtm", how="left")
    df = build_full_feature_pipeline(df).replace([np.inf, -np.inf], np.nan)
    df = add_time_keys(df)

    targets = config["targets"]
    answer = df[targets].copy()
    ct = pd.read_csv(Path(scada_dir) / "clean_target.csv", encoding="utf-8-sig")
    ct["forecast_kst_dtm"] = pd.to_datetime(ct.pop("kst_dtm"))
    df = df.merge(ct, on="forecast_kst_dtm", how="left")
    scale = {}
    for g in targets:
        cap = CAPACITY_KWH[g]
        tgt = df[f"{g}_cf"].to_numpy(float) * cap
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * cap)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * cap)
    return df, answer, scale


def q_params(config, quick, device):
    p = dict(config.get("model_params", {}))
    for k in ["objective", "eval_metric", "tweedie_variance_power", "early_stopping_rounds"]:
        p.pop(k, None)
    p["objective"] = "reg:quantileerror"
    p["quantile_alpha"] = ALPHAS.tolist()
    if quick:
        p["n_estimators"] = 300
    if device:
        p["device"] = device
    return p


def fit_predict_q(params, X_tr, y_tr, fd_tr, X_va, es_rounds=30, seed=0):
    """블록 단위 내부 ES 로 한 번 학습해 검증행 분위 (n, 5) 를 반환."""
    from xgboost import XGBRegressor
    tr_m, es_m = inner_es_split(fd_tr, frac=0.15, seed=seed)
    p = dict(params); p["early_stopping_rounds"] = es_rounds
    m = XGBRegressor(**p)
    m.fit(X_tr.iloc[tr_m], y_tr.iloc[tr_m],
          eval_set=[(X_tr.iloc[es_m], y_tr.iloc[es_m])], verbose=False)
    try:
        m.get_booster().set_param({"device": "cpu"}); m.set_params(device="cpu")
    except Exception:
        pass
    q = np.asarray(m.predict(X_va))
    if q.ndim == 1:                                        # 방어: (n*5,) 로 오는 경우
        q = q.reshape(len(X_va), len(ALPHAS))
    return np.sort(q, axis=1), m                           # 분위 교차 방지


# ------------------------------------------------------------------ 배치
def place_rows(qmat, cap, lam, mix=None, chunk=4096):
    """
    qmat: (n, 5) 원단위 분위.  반환: (n,) 기대효용 최대 점예측.
    분포 근사: 분위 5점의 조각선형 역CDF, 꼬리는 q05/q95 로 고정(보수적).
    후보 f = 표본점 자신 (분포가 있는 곳에만 후보를 둠).

    mix = (levels, probs) : **정지 혼합** (1차 실행 실패의 수정, §3.29)
      분위 모델은 정제 타깃(무정지 발전량)을 학습했으므로 예측 분포에 **정지 왼꼬리가 없음.**
      실제 라벨은 a = (무정지 발전량) × V,  V ~ 가용률 경험분포 (§3.26: 1.0 이 85%,
      0.833 이 10%, … 터빈 양자). V 를 독립으로 곱해 실제 라벨 분포로 되돌린 뒤 적분함.
      이게 없으면 배치가 '밴드 잡을 확률' 을 과대평가해 위로 과감하게 나감.
    """
    n = len(qmat)
    U = np.linspace(0.025, 0.975, 39)
    out = np.empty(n)
    tau = 0.10 * cap
    if mix is not None:
        lv, pw = np.asarray(mix[0], float), np.asarray(mix[1], float)
        pw = pw / pw.sum()
    for lo in range(0, n, chunk):
        Q = qmat[lo:lo + chunk]                                   # (c, 5)
        S = np.stack([np.interp(U, ALPHAS, row) for row in Q])    # (c, 39) 무정지 표본
        F = S[:, ::2]                                             # (c, 20) 후보
        if mix is None:
            A = S[:, None, :]                                     # (c, 1, 39)
            W = np.full(S.shape[1], 1.0 / S.shape[1])
        else:
            A = (S[:, :, None] * lv[None, None, :]).reshape(len(S), -1)[:, None, :]
            W = (np.full(S.shape[1], 1.0 / S.shape[1])[:, None] * pw[None, :]).reshape(-1)
        f = np.clip(F, 0, cap)[:, :, None]                        # (c, 20, 1)
        e = np.abs(f - A) / cap
        pr = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
        util = (np.where(A >= tau, -np.abs(f - A) + lam * A * pr, 0.0) * W).sum(axis=2)
        out[lo:lo + chunk] = np.clip(F[np.arange(len(F)), util.argmax(axis=1)], 0, cap)
    return out


def avail_mixture(scada_dir, g):
    """clean_target.csv 의 avail_frac 경험분포 (터빈 양자). §3.26 측정의 재사용."""
    ct = pd.read_csv(Path(scada_dir) / "clean_target.csv", encoding="utf-8-sig")
    af = ct[f"{g}_avail_frac"].dropna().round(3)
    vc = af.value_counts(normalize=True).sort_index(ascending=False)
    vc = vc[vc >= 0.002]                                          # 0.2% 미만 꼬리는 버림
    return vc.index.to_numpy(float), vc.to_numpy(float)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--v13-dir", default="./saved_models/v13")
    ap.add_argument("--save-dir", default=None, help="기본값: config 의 save_model_dir")
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--device", default=None, help="quantileerror 가 GPU 에서 죽으면 cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="저장된 분위 OOF 재사용")
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--submit-kind", default="median", choices=["median", "placed"],
                    help="median = q50 + 표준 후처리 (v17 제출 후보) / placed = 기대효용 배치")
    ap.add_argument("--log", action="store_true",
                    help="experiment_log.xlsx 에 이 버전을 등록 (중첩 후처리 기준, train.py 관례)")
    ap.add_argument("--mix-avail", action="store_true",
                    help="배치 적분에 정지 혼합(§3.26 가용률 경험분포)을 넣음 — 2차 실행용")
    args = ap.parse_args()

    t0 = time.time()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(config["seed"])
    targets = config["targets"]
    ver = config.get("version", "v17")
    qa = config.get("model_params", {}).get("quantile_alpha")
    if qa:                                        # config 가 분위를 정의하면 그것을 따름
        global ALPHAS
        ALPHAS = np.array(qa, float)
    v13 = Path(args.v13_dir)
    sdir = Path(args.save_dir or config["data_paths"]["save_model_dir"])
    sdir.mkdir(parents=True, exist_ok=True)
    joint_groups = json.load(open(v13 / "joint_groups.json")) if (v13 / "joint_groups.json").exists() else []
    print(BAR)
    print(f"step30 — B-2b 분위회귀+기대효용   분위 {ALPHAS.tolist()}   통합 {joint_groups}")

    print("🔄 데이터 조립 (train.py 미러, clean-target)...")
    df, answer, scale = assemble(config, args.scada_dir)
    feature_cols = joblib.load(v13 / "feature_cols.pkl")
    params = q_params(config, args.quick, args.device)

    # λ 상수 (그룹별): 학습 라벨 채점행 평균 발전량
    lam = {}
    for g in targets:
        a = answer[g].to_numpy(float); cap = CAPACITY_KWH[g]
        sc = a[np.isfinite(a) & (a >= 0.10 * cap)]
        lam[g] = cap / (4.0 * sc.mean())
        print(f"   {g.replace('kpx_group_','G')}: Ā {sc.mean():,.0f} kWh  λ {lam[g]:.3f}")

    # ------------------------------------------------------------ 분위 OOF
    qpath = sdir / "oof_quantiles.npz"
    if args.skip_train and qpath.exists():
        z = np.load(qpath)
        oof_q = {g: z[g] for g in targets}
        print("♻️  저장된 분위 OOF 재사용")
    else:
        oof_q = {g: np.full((len(df), len(ALPHAS)), np.nan) for g in targets}
        yrs_all, fds_all = df["_year"].to_numpy(), df["_fday"].to_numpy()

        # --- 통합 (G1·G3) : long-format, v13 폴드 피처 재사용
        if joint_groups:
            long = build_long(df, feature_cols, targets, weight_col=None)
            fm = np.isfinite(long["_y"].to_numpy())
            lyr, lfd = long["_year"].to_numpy(), long["_fday"].to_numpy()
            for tr, va, name in make_folds(lyr, lfd, scheme="loyo", seed=config["seed"]):
                cols = joblib.load(v13 / f"featcols_joint_{name}.pkl")
                trf = tr[fm[tr]]
                print(f"   🔗 joint {name}: fit {len(trf):,}행 × {len(cols)}피처")
                q, _ = fit_predict_q(params, long[cols].iloc[trf], long["_y"].iloc[trf],
                                     lfd[trf], long[cols].iloc[va],
                                     es_rounds=config.get("early_stopping_rounds", 30),
                                     seed=config["seed"])
                sub = long.iloc[va]
                for g in joint_groups:
                    mk = (sub["_gname"] == g).to_numpy()
                    oof_q[g][sub["_row"].to_numpy()[mk]] = np.clip(q[mk], 0, 1.15 * CREF)

        # --- 독립 (나머지 = G2)
        for g in [t for t in targets if t not in joint_groups]:
            fitm = df[g].notna().to_numpy()
            cap = CAPACITY_KWH[g]
            for tr, va, name in make_folds(yrs_all, fds_all, scheme="loyo", seed=config["seed"]):
                cols = joblib.load(v13 / f"featcols_{g}_{name}.pkl")
                trf = tr[fitm[tr]]
                if len(trf) < 100:
                    continue
                print(f"   🌀 {g.replace('kpx_group_','G')} {name}: fit {len(trf):,}행 × {len(cols)}피처")
                q, _ = fit_predict_q(params, df[cols].iloc[trf], df[g].iloc[trf],
                                     fds_all[trf], df[cols].iloc[va],
                                     es_rounds=config.get("early_stopping_rounds", 30),
                                     seed=config["seed"])
                oof_q[g][va] = np.clip(q, 0, 1.15 * cap)
        np.savez_compressed(qpath, **oof_q)
        print(f"💾 분위 OOF 저장: {qpath}")

    # scale(k) 반영: cf 스케일 -> 실제 kWh 스케일
    for g in targets:
        oof_q[g] = oof_q[g] * scale[g]

    # ------------------------------------------------------------ 3원 평가
    print(BAR)
    print("평가 — [v13 후처리] vs [분위중앙값+후처리] vs [placed]  (전부 같은 행)")
    v13_oof = pd.read_csv(v13 / "oof_preds.csv", index_col=0) \
                .reindex(index=df.index, columns=targets).astype(float)
    med = pd.DataFrame({g: np.clip(oof_q[g][:, 2], 0, CAPACITY_KWH[g]) for g in targets},
                       index=df.index)

    print("\n[분위 캘리브레이션 — 채점행 실측 P(a <= q_j). 목표 = alpha. 크게 어긋나면 배치 적분이 무의미]")
    print("  " + f"{'그룹':<5}" + "".join(f"{a:>8.2f}" for a in ALPHAS))
    for g in targets:
        a = answer[g].to_numpy(float); Q = oof_q[g]; cap = CAPACITY_KWH[g]
        m = np.isfinite(a) & np.isfinite(Q).all(1) & (a >= 0.10 * cap)
        cov = (a[m, None] <= Q[m]).mean(0)
        print("  " + f"{g.replace('kpx_group_','G'):<5}" + "".join(f"{c:>8.2f}" for c in cov))

    mixes = {g: (avail_mixture(args.scada_dir, g) if args.mix_avail else None) for g in targets}
    if args.mix_avail:
        for g in targets:
            lv, pw = mixes[g]
            print(f"  정지 혼합 {g.replace('kpx_group_','G')}: " +
                  "  ".join(f"{l:.2f}:{w*100:.0f}%" for l, w in zip(lv[:4], pw[:4])))
    placed = pd.DataFrame({g: place_rows(oof_q[g], CAPACITY_KWH[g], lam[g], mix=mixes[g])
                           for g in targets}, index=df.index)
    placed.to_csv(sdir / "oof_placed.csv"); med.to_csv(sdir / "oof_median.csv")

    print("\n[게이트 1 — 정확도 비열등성: 분위 중앙값 vs v13 raw]")
    r1 = is_difference_real(df, answer, med, v13_oof, targets, name_a="q50", name_b="v13raw")
    stop = bool(r1 and r1["real"] and r1["mean"] < 0)
    if stop:
        print("   🛑 3/3 일관 악화 — 분위 학습이 정확도를 깎음. 여기서 중단, 배치 판정 안 함.")

    pp13 = optimize_postprocessing(answer, v13_oof, mode="piecewise", verbose=False)
    v13_post = apply_postprocessing(v13_oof.copy(), pp13)
    ppq = optimize_postprocessing(answer, med, mode="piecewise", verbose=False)
    med_post = apply_postprocessing(med.copy(), ppq)

    rows = [("v13 raw", v13_oof), ("v13 + 후처리(insample)", v13_post),
            ("q50 raw", med), ("q50 + 후처리(insample)", med_post), ("placed", placed)]
    print(f"\n  {'구성':<26}{'total':>9}{'1-NMAE':>9}{'FICR':>9}")
    for nm, o in rows:
        s = total_score(answer, o, targets)
        print(f"  {nm:<26}{s[0]:9.4f}{s[1]:9.4f}{s[2]:9.4f}")

    if not stop:
        print("\n[게이트 2 — 배치 이득 (참고. 최종은 LB): placed vs v13+후처리]")
        r2 = is_difference_real(df, answer, placed, v13_post, targets,
                                name_a="placed", name_b="v13post")
        gpos, gl = 0, []
        for g in targets:
            cap = CAPACITY_KWH[g]; a = answer[g].to_numpy(float)
            d = (group_score(a, placed[g].to_numpy(float), cap)[0]
                 - group_score(a, v13_post[g].to_numpy(float), cap)[0])
            gpos += int(np.isfinite(d) and d > 0); gl.append(f"{g.replace('kpx_group_','G')} {d:+.4f}")
        ok = bool(r2 and r2["real"] and r2["mean"] > 0 and gpos >= 2)
        print("   그룹별: " + "  ".join(gl) + f"  (양수 {gpos}/3)")
        print(f"   => {'✅ 제출 후보 (LB 로 최종 판정)' if ok else '❌ 채택 안 함'}")
        print("\n   분해: [q50+후처리] − [v13+후처리] = 분위 학습의 몫, "
              "[placed] − [q50+후처리] = 배치의 몫")

    if args.log:
        from src.validation import nested_postprocess_score
        from src.logger import log_experiment
        print("\n📐 로그 등록용 중첩 후처리 (train.py 관례와 동일 기준):")
        _, raw_s, nst = nested_postprocess_score(df, answer, med, targets,
                                                 optimize_postprocessing, apply_postprocessing,
                                                 mode="piecewise", verbose=True)
        try:
            log_experiment(config=config, total_score=nst[0], one_minus_nmae=nst[1], ficr=nst[2],
                           validation="loyo", target_kind="clean_cf", es_mode="inner",
                           post_mode="piecewise", objective="reg:quantileerror(q50)",
                           raw_oof=raw_s[0],
                           n_features=200,
                           features_summary="v13 미러 + 손실만 분위회귀(5분위, 점예측 q50)",
                           notes="step30/§3.29: q50-v13 raw 짝비교 +0.0131 (3/3, FICR만 +0.0239). "
                                 "v14 교란(W) 분리 시험")
        except Exception as e:
            print(f"   ⚠ 로그 기록 실패({type(e).__name__}) — 수동 등록 필요")

    # ------------------------------------------------------------ 제출
    if args.make_submission:
        print(BAR); print("제출 파일 생성 — 테스트 분위 예측 후 배치")
        te = Path(config["data_paths"]["test_dir"])
        cands = [Path("./sample_submission.csv"), te / "sample_submission.csv",
                 Path("./data/sample_submission.csv")]
        sp = next((c for c in cands if c.exists()), None) or list(Path(".").rglob("sample_submission.csv"))[0]
        sample = pd.read_csv(sp, encoding="utf-8-sig")
        sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
        ldaps = pd.read_csv(te / "ldaps_test.csv", encoding="utf-8-sig")
        gfs = pd.read_csv(te / "gfs_test.csv", encoding="utf-8-sig")
        w = process_weather_data(ldaps, "ldaps").merge(
            process_weather_data(gfs, "gfs"), on="forecast_kst_dtm",
            how="inner", suffixes=("", "_gfsdup"))
        w = w.drop(columns=[c for c in w.columns if c.endswith("_gfsdup")])
        tdf = sample[["forecast_id", "forecast_kst_dtm"]].merge(w, on="forecast_kst_dtm", how="left")
        X = build_full_feature_pipeline(tdf).replace([np.inf, -np.inf], np.nan)
        X = add_time_keys(X) if "data_available_kst_dtm" in X.columns else X

        print("   전체 학습 데이터로 재학습 (LOYO 아님, 배포용)...")
        sub = sample[["forecast_id", "forecast_kst_dtm"]].copy()
        yrs_all, fds_all = df["_year"].to_numpy(), df["_fday"].to_numpy()
        if joint_groups:
            long = build_long(df, feature_cols, targets, weight_col=None)
            fm = np.isfinite(long["_y"].to_numpy())
            cols = joblib.load(v13 / "featcols_joint_val2024.pkl")
            Xl = long[cols]
            longtest = {}
            for g in joint_groups:
                sp_ = GROUP_SPEC[g]
                d = X.reindex(columns=feature_cols).astype(np.float32).copy()
                for k in range(3):
                    d[f"grp_{k}"] = np.float32(1.0 if sp_["gid"] == k else 0.0)
                d["grp_is_vestas"] = np.float32(sp_["is_vestas"])
                d["grp_n_turb"] = np.float32(sp_["n_turb"])
                d["grp_rotor_d"] = np.float32(sp_["rotor_d"])
                longtest[g] = d[cols]
            q, m = fit_predict_q(params, Xl[fm], long["_y"][fm],
                                 long["_fday"].to_numpy()[fm], Xl[fm].iloc[:1])
            for g in joint_groups:
                qt = np.asarray(m.predict(longtest[g]))
                if qt.ndim == 1:
                    qt = qt.reshape(len(longtest[g]), len(ALPHAS))
                qt = np.sort(np.clip(qt, 0, 1.15 * CREF), axis=1) * scale[g]
                sub[g] = (np.clip(qt[:, 2], 0, CAPACITY_KWH[g]) if args.submit_kind == "median"
                          else place_rows(qt, CAPACITY_KWH[g], lam[g], mix=mixes.get(g)))
        for g in [t for t in targets if t not in joint_groups]:
            fitm = df[g].notna().to_numpy()
            cols = joblib.load(v13 / f"featcols_{g}_val2024.pkl")
            q, m = fit_predict_q(params, df[cols][fitm], df[g][fitm],
                                 fds_all[fitm], df[cols][fitm].iloc[:1])
            qt = np.asarray(m.predict(X.reindex(columns=cols)))
            if qt.ndim == 1:
                qt = qt.reshape(len(X), len(ALPHAS))
            qt = np.sort(np.clip(qt, 0, 1.15 * CAPACITY_KWH[g]), axis=1) * scale[g]
            sub[g] = (np.clip(qt[:, 2], 0, CAPACITY_KWH[g]) if args.submit_kind == "median"
                      else place_rows(qt, CAPACITY_KWH[g], lam[g], mix=mixes.get(g)))

        if args.submit_kind == "median":
            print("  ⚙️ q50 위에 표준 piecewise 후처리 적합(OOF 전체) 후 적용 — train.py 배포 관례와 동일")
            ppm = optimize_postprocessing(answer, med, mode="piecewise", verbose=False)
            sub = apply_postprocessing(sub, ppm)
            for g in targets:
                sub[g] = np.clip(sub[g].to_numpy(float), 0, CAPACITY_KWH[g])
        sub["forecast_kst_dtm"] = sub["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
        outdir = Path(config["data_paths"]["submission_dir"]); outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / (f"submit_{ver}.csv" if args.submit_kind == "median"
                        else f"submit_{ver}_placed.csv")
        sub.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  🚀 {out}   행 {len(sub):,}  결측 {int(sub[targets].isna().sum().sum())}")
        for g in targets:
            v = sub[g].to_numpy(float)
            print(f"     {g}: 평균 {np.mean(v)/CAPACITY_KWH[g]*100:5.1f}%cap  "
                  f"최대 {np.max(v)/CAPACITY_KWH[g]*100:5.1f}%cap")

    print(BAR); print(f"경과 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()