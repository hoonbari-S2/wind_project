"""
scripts/step24_stability.py
==============================================================================
A-5 — 대기 안정도 지표. 격자 축이 아니라 물리 축이므로 통합 학습이 흡수할 이유가 없음

왜 이 축인가 — 신호가 있다는 증거가 셋
  1. §3.4  폴드 상위 30 에 생존한 파생 피처가 **딱 2개**인데 그중 하나가
           `ldaps_blh_v117_interaction`(v/blh). 안정도 축에 신호가 있다는 직접 증거임.
  2. §3.4  `surface_0_dswrf` 가 7개 격자에서 반복 등장 — 주야/안정도 대리변수로 지목됨.
  3. EDA 셀 12  새벽 03~04시 평균 발전량 최고(G2 ~8,800 / G1 ~8,200 kWh).
           **야간 하층제트**는 안정 경계층의 전형적 현상임.

  그런데 지금 있는 것은 전부 **대리변수**뿐임 (blh, dswrf).
  벌크 리처드슨 수·온위 같은 **정식 지표가 하나도 없음.**

미활용 원본 (§8.5 가 "컬럼이 아니라 상호작용 형태가 없다" 고 지적한 그것)
      LDAPS  surface_0_NDNLW   순 하향 장파 = 야간 복사냉각 직접 지표. dswrf 보다 직접적
      LDAPS  etc_0_VLCDC       매우 낮은 층 운량 = 복사냉각을 막는 요인
      GFS    planetaryBoundaryLayer_0_VRATE   경계층 수직속도 = 혼합 강도
      GFS    isobaricInhPa_500_*  §4.4 가 "종관 흐름은 850 또는 500hPa 지위고도" 라 지시했으나
                                  500 은 파생이 하나도 없음
      GFS    isobaricInhPa_700_*  전혀 미사용

설계 원칙 (§3.2 · §3.18 · §3.21 의 교훈을 전부 반영)
  * **전부 다변수 결합.** 단일 컬럼의 단조변환은 만들지 않음 (§3.2)
  * **소수 정예 약 20개.** step23 의 54개가 선택 예산을 먹은 전례가 있음 (§3.18)
  * **처음부터 v13 통합 구성에서 측정함.** step23 이 그룹 독립에서 통과했다가 통합에서
    소멸한 것이 §3.21 의 교훈임. **배포 구성에서 재는 것이 맞음.**
  * run_cv / run_cv_joint 를 그대로 호출해 train.py 와 **절차를 100% 동일하게** 맞춤

판정 (사전 등록, §3.6 규칙 2·6)
  raw OOF 연도짝비교. 부호 3/3 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수.

사전 예측 (결과 보기 전 고정 — 이것이 판정의 절반임)
  안정도는 **격자 축이 아니므로 §3.21 의 흡수를 겪을 이유가 없음.**
  그리고 §3.20 과 달리 **G3 편중이 예상되지 않음** — 안정 경계층은 세 그룹에 공통으로 작용함.
  대신 **기전 확인**: 개선이 안정 구간(Δθ 상위)에 몰려야 함.
  안정도와 무관하게 고르게 오르면 그냥 피처 하나 더 준 것이며 기전 설명이 안 됨.

실행
    python scripts/step24_stability.py --config configs/config_v13.yaml
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
from src.validation import (quiet_warnings, add_time_keys, run_cv, run_cv_joint,
                            total_score, group_score, score_by_year, is_difference_real)

quiet_warnings()
BAR = "=" * 84
EPS = 1e-6
G = 9.80665
KAPPA = 0.2857          # R/cp
L = "ldaps_mean_"
F = "gfs_mean_"


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


def _g(df, c):
    return df[c].to_numpy(float) if c in df.columns else None


def _K(t):
    return np.where(t < 100.0, t + 273.15, t) if t is not None else None


def _ws(u, v):
    return np.sqrt(u * u + v * v) if (u is not None and v is not None) else None


def _wd(u, v):
    return np.degrees(np.arctan2(-u, -v)) % 360.0


def add_stability(df, verbose=True):
    """
    대기 안정도 지표. 전부 2개 이상 컬럼의 비선형 결합이며 트리가 만들 수 없는 형태임.

        온위        θ = T·(1e5/p)^(R/cp)
        연직 안정도 Δθ = θ(850hPa) − θ(2m)      양수면 안정(역전), 음수면 불안정
        벌크 Ri     Ri_b = (g/θ̄)·(Δθ·Δz) / (Δu)²   대기 안정도의 표준 지표
        풍향 회전   veering = sin(wd850 − wd100)     시계방향 = 온난이류 = 안정화
    """
    o = {}
    # ---- 온위와 연직 안정도 ------------------------------------------------
    t2l, spl = _K(_g(df, L + "heightAboveGround_2_t")), _g(df, L + "surface_0_sp")
    t2g, spg = _K(_g(df, F + "heightAboveGround_2_2t")), _g(df, F + "surface_0_sp")
    t850 = _K(_g(df, F + "isobaricInhPa_850_t"))
    t700 = _K(_g(df, F + "isobaricInhPa_700_t"))

    th2 = None
    if t2l is not None and spl is not None:
        o["theta_2m_ldaps"] = t2l * (1e5 / np.maximum(spl, 1.0)) ** KAPPA
    if t2g is not None and spg is not None:
        th2 = t2g * (1e5 / np.maximum(spg, 1.0)) ** KAPPA
        o["theta_2m_gfs"] = th2
    if t850 is not None:
        th850 = t850 * (1e5 / 85000.0) ** KAPPA
        o["theta_850"] = th850
        if th2 is not None:
            o["dtheta_850_2m"] = th850 - th2                      # ★ 안정도 핵심
    if t700 is not None and th2 is not None:
        o["dtheta_700_2m"] = t700 * (1e5 / 70000.0) ** KAPPA - th2

    # ---- 벌크 리처드슨 수 --------------------------------------------------
    ws10g = _ws(_g(df, F + "heightAboveGround_10_10u"), _g(df, F + "heightAboveGround_10_10v"))
    ws100 = _ws(_g(df, F + "heightAboveGround_100_100u"), _g(df, F + "heightAboveGround_100_100v"))
    ws850 = _ws(_g(df, F + "isobaricInhPa_850_u"), _g(df, F + "isobaricInhPa_850_v"))
    ws500 = _ws(_g(df, F + "isobaricInhPa_500_u"), _g(df, F + "isobaricInhPa_500_v"))
    if "dtheta_850_2m" in o and ws850 is not None and ws10g is not None and th2 is not None:
        dz = 1450.0                                               # 2m -> 850hPa 근사 두께
        du = np.abs(ws850 - ws10g) + 0.5                          # 0 나눗셈 방지
        o["bulk_richardson"] = (G / np.maximum(th2, 1.0)) * o["dtheta_850_2m"] * dz / (du ** 2)
        o["shear_2_850"] = du / dz * 1000.0                       # m/s per km

    # ---- 야간 복사냉각 (NDNLW × 운량) --------------------------------------
    nlw = _g(df, L + "surface_0_NDNLW")
    vlc = _g(df, L + "etc_0_VLCDC")
    lcc = _g(df, L + "etc_0_lcc")
    blh = _g(df, L + "etc_0_blh")
    if nlw is not None:
        cloud = vlc if vlc is not None else lcc
        if cloud is not None:
            cf = np.clip(cloud / (np.nanmax(np.abs(cloud)) + EPS), 0, 1)
            o["radcool_clearsky"] = nlw * (1.0 - cf)              # 맑은 하늘 복사냉각
        if blh is not None:
            o["radcool_per_blh"] = nlw / (blh + 1.0) * 1000.0     # 얕은 경계층에서 증폭

    # ---- 경계층 혼합 강도 --------------------------------------------------
    vrate = _g(df, F + "planetaryBoundaryLayer_0_VRATE")
    pbl = _ws(_g(df, F + "planetaryBoundaryLayer_0_u"), _g(df, F + "planetaryBoundaryLayer_0_v"))
    if vrate is not None and blh is not None:
        o["mixing_vrate_blh"] = vrate * blh / 1000.0
        o["vrate_per_blh"] = vrate / (blh + 1.0) * 1000.0
    if pbl is not None and ws100 is not None:
        o["ws100_over_pbl"] = ws100 / (pbl + EPS)

    # ---- 종관 (500hPa) — §4.4 가 지시했으나 미활용이던 것 -------------------
    if ws500 is not None and ws850 is not None:
        o["ws500_over_850"] = ws500 / (ws850 + EPS)
    if ws500 is not None and ws100 is not None:
        o["ws500_over_100"] = ws500 / (ws100 + EPS)

    # ---- 풍향 회전 (온도 이류) — 비선형이라 트리가 못 만듦 ------------------
    u850, v850 = _g(df, F + "isobaricInhPa_850_u"), _g(df, F + "isobaricInhPa_850_v")
    u100, v100 = _g(df, F + "heightAboveGround_100_100u"), _g(df, F + "heightAboveGround_100_100v")
    if all(z is not None for z in (u850, v850, u100, v100)):
        d = np.radians(_wd(u850, v850) - _wd(u100, v100))
        o["veering_sin"] = np.sin(d)                              # + = 온난이류 = 안정화
        o["veering_cos"] = np.cos(d)

    # ---- 안정도 × 풍속 상호작용 (핵심) --------------------------------------
    v117 = _g(df, "gfs_v117_powerlaw")
    if v117 is not None:
        for k in ("dtheta_850_2m", "bulk_richardson", "radcool_clearsky"):
            if k in o:
                o[f"{k}_x_v117"] = o[k] * v117

    out = pd.DataFrame(o, index=df.index).replace([np.inf, -np.inf], np.nan)
    # 벌크 Ri 는 꼬리가 매우 두꺼움 — 분위 클리핑
    if "bulk_richardson" in out:
        lo, hi = np.nanquantile(out["bulk_richardson"], [0.005, 0.995])
        out["bulk_richardson"] = out["bulk_richardson"].clip(lo, hi)
    if verbose:
        print(f"🌡️  안정도 피처 {out.shape[1]}개 생성 (전부 다변수 결합)")
        print("    " + ", ".join(list(out.columns)[:6]) + " ...")
    return out


def build_oof(df, feats, targets, cfg, scale, joint, seed, top_k):
    """train.py 와 동일 절차: run_cv 로 전 그룹 -> joint 그룹만 run_cv_joint 로 덮어씀."""
    mt, mp = cfg.get("model_type", "XGBoost"), cfg.get("model_params", {})
    esr = cfg.get("early_stopping_rounds", 30)
    oof, _, _, _ = run_cv(df, feats, targets, mt, mp, scheme="loyo",
                          es_rounds=esr, es_mode="refit", seed=seed,
                          verbose=False, top_k=top_k)
    if joint:
        oj, _, _ = run_cv_joint(df, feats, targets, mt, mp, scheme="loyo",
                                es_rounds=esr, es_mode="refit", seed=seed,
                                verbose=False, top_k=top_k)
        for g in joint:
            oof[g] = oj[g]
    for g in targets:
        oof[g] = np.clip(oof[g] * scale[g], 0, CAPACITY_KWH[g])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step24")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--joint", default="kpx_group_1,kpx_group_3")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]; seed = cfg["seed"]
    joint = [x.strip() for x in args.joint.split(",") if x.strip()]
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1  데이터 + 피처 + 정제 타깃 (v13 구성 그대로)")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = add_time_keys(build_full_feature_pipeline(
        labels.merge(w, on="forecast_kst_dtm", how="left")).replace([np.inf, -np.inf], np.nan))

    answer = df[targets].copy()
    ct = pd.read_csv(Path(args.scada_dir) / "clean_target.csv", encoding="utf-8-sig")
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

    stab = add_stability(df)
    df = pd.concat([df, stab], axis=1).replace([np.inf, -np.inf], np.nan)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets])
    allnum = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    newset = set(stab.columns)
    base = [c for c in allnum if c not in newset]
    full = base + [c for c in allnum if c in newset]
    print(f"  기존 {len(base)}  +안정도 {len(full)-len(base)}  =  {len(full)}   통합그룹 {joint}")

    print(BAR); print("STEP 2  A(기존) / B(+안정도) — 둘 다 v13 통합 구성 (§3.21 교훈)")
    oofs = {}
    for nm, fe in [("A 기존", base), ("B +안정도", full)]:
        oofs[nm] = build_oof(df, fe, targets, cfg, scale, joint, seed, args.top_k)
        s = total_score(answer, oofs[nm], targets)
        oofs[nm].to_csv(odir / f"oof_{nm.split()[0]}.csv")
        print(f"  {nm:12s} raw {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}   ({time.time()-t0:.0f}s)")

    print(BAR); print("STEP 3  판정 (§3.6 규칙 2·6)")
    for n, o in oofs.items():
        score_by_year(df, answer, o, targets, label=n)
    print()
    res = is_difference_real(df, answer, oofs["B +안정도"], oofs["A 기존"], targets,
                             name_a="B +안정도", name_b="A 기존")
    gpos, line = 0, []
    for g in targets:
        cap = CAPACITY_KWH[g]; a = answer[g].to_numpy(float)
        sa = group_score(a, oofs["A 기존"][g].to_numpy(float), cap)[0]
        sb = group_score(a, oofs["B +안정도"][g].to_numpy(float), cap)[0]
        gpos += int(np.isfinite(sb - sa) and sb > sa)
        line.append(f"{g.replace('kpx_','')} {sb-sa:+.4f}")
    ok = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2)
    print("  그룹별: " + "  ".join(line) + f"   (양수 {gpos}/{len(targets)})")
    print(f"  => {'✅ 채택 후보' if ok else '❌ 채택 안 함'}")

    print(BAR); print("STEP 4  기전 확인 — 개선이 안정 구간에 몰리는가")
    print("  Δθ(850hPa − 2m 온위) 5분위. 양수·큼 = 안정(역전). 개선이 상위 분위에 몰려야 기전이 맞음.\n")
    key = "dtheta_850_2m" if "dtheta_850_2m" in df.columns else None
    if key is None:
        print("  (Δθ 계산 불가 — 850hPa 기온 컬럼 없음)")
    else:
        dth = df[key].to_numpy(float)
        q = np.nanquantile(dth[np.isfinite(dth)], np.linspace(0, 1, 6)); q[0], q[-1] = -np.inf, np.inf
        print("    " + f"{'Δθ 분위':<14s}{'평균 Δθ':>10s}{'채점행':>9s}{'A':>9s}{'B':>9s}{'Δ':>9s}")
        for i in range(5):
            m0 = np.isfinite(dth) & (dth >= q[i]) & (dth < q[i + 1])
            sa_l, sb_l, n = [], [], 0
            for g in targets:
                cap = CAPACITY_KWH[g]; a = answer[g].to_numpy(float)
                mm = m0 & np.isfinite(a) & (a >= 0.10 * cap)
                if mm.sum() < 200:
                    continue
                n += int(mm.sum())
                sa_l.append(group_score(a[mm], oofs["A 기존"][g].to_numpy(float)[mm], cap)[0])
                sb_l.append(group_score(a[mm], oofs["B +안정도"][g].to_numpy(float)[mm], cap)[0])
            if not sa_l:
                continue
            lab = f"Q{i+1}" + (" 최불안정" if i == 0 else " 최안정" if i == 4 else "")
            print("    " + f"{lab:<14s}{np.nanmean(dth[m0]):10.2f}{n:9d}"
                  f"{np.mean(sa_l):9.4f}{np.mean(sb_l):9.4f}{np.mean(sb_l)-np.mean(sa_l):+9.4f}")
        print("\n  개선이 Q4·Q5(안정)에 몰리면 기전 확인. 고르게 퍼지면 '그냥 피처 하나 더' 이며")
        print("  기전 설명이 안 되므로 채택하더라도 A-5 확장 근거로는 쓰지 않음.")

    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")
    print("사전 예측: 안정도는 격자 축이 아니므로 §3.21 의 통합 흡수를 겪을 이유가 없음.")
    print("           G3 편중도 예상되지 않음 — 안정 경계층은 세 그룹에 공통 작용함.")


if __name__ == "__main__":
    main()