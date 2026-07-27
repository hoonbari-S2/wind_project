"""
scripts/step18_block_features.py
==============================================================================
A-6 — 아직 한 번도 안 준 축: lead time 과 예보 블록 내부 구조

왜 지금인가
  후처리 축은 닫혔다. step15/16 이 양방향으로 닫았다(§8.4 확정).
  남은 몫은 정확도 트랙이고(총점 0.0214), §3.8 따름정리에 따라
  '기존 컬럼의 재계산' 은 전부 정보가 0 이다. 그런데 A-6 은 재계산이 아니다.
  **모델이 아직 한 번도 못 본 두 가지**를 준다.

  (1) lead_hours — 예보 블록은 매일 09KST 초기화 -> 13KST 공개 -> 익일 01시~익익일 00시.
      즉 같은 예보 안에서도 lead time 이 12~35시간으로 3배 차이 난다.
      NWP 정확도는 lead time 에 따라 크게 떨어지는데 **모델은 그 축을 모른다.**
      §6: 위반 아니다. forecast_kst_dtm 과 data_available_kst_dtm 모두 예측기준시점에
      손에 있고 주최측이 test CSV 에도 준다.

  (2) 블록 내 전방 참조 — 24시간이 한 덩어리로 동시 공개되므로
      groupby('data_available_kst_dtm') 안에서 shift(-1) / rolling(center=True) 는 합법이다
      (§6, causality.py 로 실행 가능한 증명이 이미 있다).
      현재 features.py 의 gfs_v117_lag1 / ramp_1h / roll_mean_3h 는 **groupby 없이**
      계산돼 블록 경계를 넘는다. 위반은 아니지만(후방 참조) 서로 다른 초기화 시각을
      섞고 있어 물리적으로 부정확하다. 이 실험은 그 수정도 겸한다.

  §3.5 분류로는 **정보형 변화**다. 관측된 전이율 79%.

설계 — 2x2 요인 (step9 와 같은 구조. 주효과를 분리해야 원인 귀속이 된다)
    A  v13 그대로                     (기준)
    B  + lead_hours 계열만            (피처 3개)
    C  + 블록 내부 구조 피처만
    D  B + C
  나머지는 전부 v13 고정: 정제 타깃 / 폴드별 top-200 / LOYO / Tweedie.
  통합(long-format)은 쓰지 않고 그룹 독립으로 돌린다 — 조건 4개를 빠르게 비교하기 위함.
  기준선도 같은 조건(독립)이라 짝비교는 공정하다. 채택되면 통합을 얹어 재측정한다.

블록 피처를 어느 컬럼에 걸 것인가
  812개 전부에 6종 연산을 걸면 4,800개가 되어 §3.3 의 희석이 다시 커진다.
  §3.4 의 생존 피처 패턴(U성분 / v117 / wpd / blh / dswrf / gust)으로 기저 컬럼을
  고정 규칙으로 고른다. 고정 규칙이라 폴드를 보지 않는다 = 선택 누설 없음.

판정 (사전 등록 — 결과 보기 전 고정)
  raw OOF 연도짝비교. 후처리 점수는 판정에 쓰지 않는다 (§3.6 규칙 1).
  채택: 부호 3/3 일치 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수.
  D 가 통과하면 B/C 주효과로 어느 쪽이 원인인지 적고, 큰 쪽만 v15 에 넣는다.
  (§3.5 규칙: 무엇이 원인인지 모른 채 두 개를 같이 바꾸면 다음 실험을 설계할 수 없다)

실행
    python scripts/step18_block_features.py --config configs/config_v13.yaml
    python scripts/step18_block_features.py --config configs/config_v13.yaml --audit
==============================================================================
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_full_feature_pipeline
from src.utils import seed_everything, CAPACITY_KWH
from src.validation import (quiet_warnings, add_time_keys, make_folds, fit_no_leak,
                            total_score, group_score, score_by_year, is_difference_real)

quiet_warnings()
BAR = "=" * 78
CREF = 21600.0
AVAIL = "data_available_kst_dtm"

# 블록 연산을 걸 기저 컬럼 패턴 — §3.4 의 생존 피처 계열. 고정 규칙(폴드 미참조).
BASE_PATTERNS = ["_mean_heightAboveGround_10_10u", "_mean_heightAboveGround_100_100u",
                 "gfs_v117_powerlaw", "wpd_ldaps_10m", "wpd_gfs_117m",
                 "ldaps_blh_v117_interaction", "ldaps_mean_etc_0_blh",
                 "gfs_mean_surface_0_gust", "ldaps_sw_total",
                 "ldaps_mean_heightAboveGround_2_t", "gfs_alpha_shear"]
MAX_BASE = 14


def process_weather_data(df, prefix):
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    idc = {"forecast_kst_dtm", "grid_id", AVAIL}
    vc = [c for c in df.columns if c not in (idc | {"latitude", "longitude"})]
    piv = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=vc)
    piv.columns = [f"{prefix}_g{c[1]}_{c[0]}" for c in piv.columns]
    agg = df.groupby("forecast_kst_dtm")[vc].mean()
    agg.columns = [f"{prefix}_mean_{c}" for c in agg.columns]
    out = piv.reset_index().merge(agg.reset_index(), on="forecast_kst_dtm", how="inner")
    if AVAIL in df.columns:
        av = df.groupby("forecast_kst_dtm")[AVAIL].first().reset_index()
        out = out.merge(av, on="forecast_kst_dtm", how="left")
    return out


def pick_base_cols(df):
    cols = []
    for pat in BASE_PATTERNS:
        hit = [c for c in df.columns if pat in c and df[c].dtype != object]
        cols += sorted(hit)[:2]                      # 패턴당 최대 2개, 알파벳 순 (결정적)
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out[:MAX_BASE]


def add_lead_features(df):
    """lead time 축. 예보 정확도가 이 축을 따라 떨어지는데 모델이 모르고 있다."""
    t = pd.to_datetime(df["forecast_kst_dtm"])
    av = pd.to_datetime(df[AVAIL])
    lead = (t - av).dt.total_seconds() / 3600.0
    out = pd.DataFrame(index=df.index)
    out["lead_hours"] = lead.to_numpy()
    out["lead_frac"] = ((lead - lead.min()) / max(lead.max() - lead.min(), 1e-9)).to_numpy()
    # 블록 안에서 몇 번째 시각인가 (0~23). lead 와 거의 같지만 결측 블록에서 갈린다
    out["blk_pos"] = df.groupby(av, sort=False).cumcount().to_numpy().astype(float)
    return out


def add_block_features(df, base_cols):
    """
    블록 경계를 절대 넘지 않는 전/후방 참조. causality.py 의 safe_block_ops 와 동일 원리.
    24시간이 동시 공개이므로 블록 안에서는 전방 참조도 합법이다 (§6).
    """
    gb = df.groupby(pd.to_datetime(df[AVAIL]), sort=False)
    new = {}
    for c in base_cols:
        s = df[c]
        new[f"{c}__blag1"] = gb[c].shift(1)
        new[f"{c}__blead1"] = gb[c].shift(-1)                       # <- 새 축
        new[f"{c}__blead2"] = gb[c].shift(-2)                       # <- 새 축
        new[f"{c}__bcwin3"] = gb[c].transform(
            lambda x: x.rolling(3, center=True, min_periods=1).mean())
        bm = gb[c].transform("mean")
        new[f"{c}__bmean"] = bm
        new[f"{c}__bstd"] = gb[c].transform("std")
        new[f"{c}__banom"] = s - bm                                 # 그날 대비 편차
        new[f"{c}__bramp"] = new[f"{c}__blead1"] - new[f"{c}__blag1"]   # 블록 내 중심차분
    return pd.DataFrame(new, index=df.index)


def run_arm(df, feats, targets, folds, fds, scale, answer, mtype, mparams, esr, sd, top_k):
    oof = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    for g in targets:
        fm = df[g].notna().to_numpy()
        pred = np.full(len(df), np.nan)
        for tr, va, _ in folds:
            trf = tr[fm[tr]]
            if len(trf) < 300:
                continue
            m0, _ = fit_no_leak(mtype, mparams, df[feats].iloc[trf], df[g].iloc[trf],
                                fds[trf], es_rounds=esr, mode="refit", seed=sd)
            imp = pd.Series(m0.feature_importances_, index=feats)
            cols = list(imp.sort_values(ascending=False).head(top_k).index)
            m, _ = fit_no_leak(mtype, mparams, df[cols].iloc[trf], df[g].iloc[trf], fds[trf],
                               es_rounds=esr, mode="refit", seed=sd)
            pred[va] = np.clip(m.predict(df[cols].iloc[va]), 0, 1.15 * CREF)
        oof[g] = np.clip(pred * scale[g], 0, CAPACITY_KWH[g])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step18")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--audit", action="store_true", help="규칙 3항 인과성 감사 (느림, 권장)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]
    sd = cfg["seed"]
    esr = cfg.get("early_stopping_rounds", 30)
    mtype = cfg.get("model_type", "XGBoost")
    mparams = cfg.get("model_params", {})
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1  데이터 + 피처 + 정제 타깃")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    raw = labels.merge(w, on="forecast_kst_dtm", how="left")
    if AVAIL not in raw.columns:
        raise KeyError(f"{AVAIL} 이 없다. process_weather_data 에서 살려야 A-6 이 성립한다.")
    df = add_time_keys(build_full_feature_pipeline(raw).replace([np.inf, -np.inf], np.nan))

    answer = df[targets].copy()
    ct = pd.read_csv(Path(args.scada_dir) / "clean_target.csv", encoding="utf-8-sig")
    ct["forecast_kst_dtm"] = pd.to_datetime(ct.pop("kst_dtm"))
    df = df.merge(ct, on="forecast_kst_dtm", how="left")
    scale = {}
    for g in targets:
        tgt = df[f"{g}_cf"].to_numpy(float) * CREF
        lab = answer[g].to_numpy(float)
        ok = np.isfinite(tgt) & np.isfinite(lab) & (tgt > 0.05 * CREF)
        scale[g] = float(np.nanmedian(lab[ok] / tgt[ok]))
        df[g] = np.clip(tgt, 0, 1.15 * CREF)

    # ---- 블록 구조 확인 + 신규 피처 ----
    blk = df.groupby(pd.to_datetime(df[AVAIL])).size()
    print(f"  블록 {len(blk)}개  블록당 {blk.min()}~{blk.max()}행 (최빈 {int(blk.mode()[0])})")
    base_cols = pick_base_cols(df)
    lead_df = add_lead_features(df)
    blk_df = add_block_features(df, base_cols)
    df = pd.concat([df, lead_df, blk_df], axis=1).replace([np.inf, -np.inf], np.nan)
    print(f"  lead 피처 {lead_df.shape[1]}개  |  블록 피처 {blk_df.shape[1]}개 "
          f"(기저 {len(base_cols)}컬럼 x 8연산)")
    print(f"  lead_hours 범위 {lead_df['lead_hours'].min():.0f} ~ {lead_df['lead_hours'].max():.0f} 시간")
    print("  기저 컬럼: " + ", ".join(c[:34] for c in base_cols[:6]) + " ...")

    excl = set(["forecast_kst_dtm", "kst_dtm", AVAIL, "year_month", "_year", "_fday"]
               + targets + [f"{g}_cf" for g in targets] + [f"{g}_avail_frac" for g in targets])
    all_num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    lead_cols = [c for c in lead_df.columns]
    blk_cols = [c for c in blk_df.columns]
    base_feats = [c for c in all_num if c not in set(lead_cols) | set(blk_cols)]

    ARMS = {"A 기준(v13)": base_feats,
            "B +lead": base_feats + lead_cols,
            "C +블록": base_feats + blk_cols,
            "D +둘다": base_feats + lead_cols + blk_cols}
    print(f"  피처 수: " + "  ".join(f"{k} {len(v)}" for k, v in ARMS.items()))

    if args.audit:
        print(BAR); print("규칙 3항 인과성 감사 — 블록 전방참조가 정말 합법인지 실행으로 증명")
        from src.causality import assert_block_structure, audit_causality
        assert_block_structure(raw)

        def fn(d):
            d2 = add_time_keys(build_full_feature_pipeline(d))
            return pd.concat([d2, add_lead_features(d2),
                              add_block_features(d2, pick_base_cols(d2))], axis=1)
        ok, rep = audit_causality(raw, fn, n_blocks=5)
        if not ok:
            print("  ❌ 위반이 나왔다. 아래 피처를 빼거나 groupby 를 다시 볼 것.")
            print(rep.head(10).to_string(index=False))
            return
        print("  ✅ 통과 — 2차 산출물에 이 로그를 첨부한다")

    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
    folds = make_folds(yrs, fds, scheme="loyo")

    print(BAR); print("STEP 2  4개 조건 학습 (그룹 독립, 나머지 v13 고정)")
    oofs = {}
    for name, feats in ARMS.items():
        oofs[name] = run_arm(df, feats, targets, folds, fds, scale, answer,
                             mtype, mparams, esr, sd, args.top_k)
        s = total_score(answer, oofs[name], targets)
        oofs[name].to_csv(odir / f"oof_{name.split()[0]}.csv")
        print(f"  {name:14s} raw {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}   ({time.time()-t0:.0f}s)")

    print(BAR); print("STEP 3  주효과 분해 (2x2)")
    v = {k: total_score(answer, o, targets)[0] for k, o in oofs.items()}
    A, B, C, D = (v["A 기준(v13)"], v["B +lead"], v["C +블록"], v["D +둘다"])
    print(f"  lead 주효과  ((B-A)+(D-C))/2 = {((B-A)+(D-C))/2:+.4f}")
    print(f"  블록 주효과  ((C-A)+(D-B))/2 = {((C-A)+(D-B))/2:+.4f}")
    print(f"  상호작용     (D-B-C+A)       = {D-B-C+A:+.4f}")

    print(BAR); print("STEP 4  판정 — raw OOF 연도짝비교 (§3.6 규칙 1~2)")
    print("  [연도별 raw]")
    for k, o in oofs.items():
        score_by_year(df, answer, o, targets, label=k)
    for k in ["B +lead", "C +블록", "D +둘다"]:
        print(f"\n  [{k} vs A]")
        res = is_difference_real(df, answer, oofs[k], oofs["A 기준(v13)"], targets,
                                 name_a=k, name_b="A")
        gpos = 0
        line = []
        for g in targets:
            cap = CAPACITY_KWH[g]
            a = answer[g].to_numpy(float)
            ga = group_score(a, oofs["A 기준(v13)"][g].to_numpy(float), cap)[0]
            gk = group_score(a, oofs[k][g].to_numpy(float), cap)[0]
            gpos += int(np.isfinite(gk - ga) and gk > ga)
            line.append(f"{g.replace('kpx_','')} {gk-ga:+.4f}")
        ok = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2)
        print("   그룹별: " + "  ".join(line) + f"   (양수 {gpos}/{len(targets)})")
        print(f"   => {'✅ 채택 후보' if ok else '❌ 채택 안 함'}")

    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")
    print("통과하면 주효과가 큰 쪽만 골라 configs/config_v15.yaml 로 본학습(통합 포함) 후 제출.")


if __name__ == "__main__":
    main()