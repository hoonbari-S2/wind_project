"""
scripts/step23_grid_dispersion.py
==============================================================================
H1 — 격자 간 산포·경도 피처. 트리가 원리적으로 만들 수 없는 양임

착상 (§3.20 진단의 직접 연장)
  process_weather_data 가 만드는 것은 두 종류뿐임.
      {prefix}_g{id}_{var}   개별 격자 (LDAPS 16 / GFS 9)
      {prefix}_mean_{var}    균등평균 1개
  **격자 16개의 표준편차·범위·공간 경도가 하나도 없음.**

  트리는 축 정렬 분할만 하므로 16개 컬럼의 표준편차를 만들 수 없음.
  (그러려면 16차원 동시 조건이 필요한데 표현 불가능함)
  => §3.2 가 허용하는 '다변수 결합' 이며, §3.8 따름정리에도 걸리지 않음.
     NWP 로부터 유도 가능하긴 하나 **트리가 실제로 유도할 수 없는** 형태임.

왜 이 축인가 — 근거 세 갈래가 같은 곳을 가리킴
  1. §3.4  개별 격자 > `_mean_` (gfs_g5_100u 12/12 vs gfs_mean_100u 10/12).
           균등평균이 정보를 죽이고 있음. 그런데 '얼마나 죽이고 있는지' 를 나타내는 양이 없음.
  2. §4.4  동일 시각 10m U성분이 G1 5.378 / G3 6.072 / 균등평균 5.298.
           그룹 간 실제 차이 13% 를 버리는 중임.
  3. §3.20 G3 는 공간 범위가 1.7배 넓어(2.05 km vs 1.20 km) 균등평균의 대표성이 가장 낮고,
           고풍속에서 `dP/dv` 가 최대라 그 오차가 최대로 증폭됨.

  `격자 std` 가 크다 = **"지금 이 시각은 균등평균을 믿으면 안 된다"** 는 신호임.
  모델이 그 신호를 받으면 공간 대표성이 낮은 시각을 다르게 취급할 수 있음.

만드는 것 (전부 다변수 결합)
  기저 변수마다: `_gstd`(격자 표준편차) `_grange`(최대−최소)
                `_gdx`(동서 경도) `_gdy`(남북 경도)   <- 격자 위경도로 최소제곱
  풍속에는 `_gcv`(std/평균) 추가. 그리고 산포×풍속수준 상호작용 소수.
  공간 경도는 **능선 가속을 직접 나타내는 양**이며 역시 트리가 만들 수 없음.

선택 예산 동시 처리 (§3.18 의 교훈)
  step18 에서 812 에 96 을 더하면서 top-200 을 그대로 둬 **기존 피처의 11% 를 잘라냈고**
  그 대가로 −0.0035 를 봤음. 같은 실수를 반복하지 않기 위해 2x2 요인으로 설계함.

      A  기존 812 + top-200      (= 기준선)
      B  +격자산포 + top-200
      C  기존 812 + top-250
      D  +격자산포 + top-250

  주효과: 격자산포 = ((B−A)+(D−C))/2 ,  선택예산 = ((C−A)+(D−B))/2
  통과하면 `step4_prune_sweep.py` 를 신규 피처 포함 상태로 다시 돌려 최적 K 를 재확정함.

판정 (사전 등록, §3.6 규칙 6)
  raw OOF 연도짝비교. 부호 3/3 일치 ∧ **양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수.
  넷 다여야 제출 1회.
  추가 진단(판정 아님): 폴드별 top-K 안에 신규 피처가 몇 개 뽑혔는지 출력함.
    많이 뽑혔는데 점수가 나쁘면 §3.18 의 선택 변위, 아예 안 뽑혔으면 그냥 무정보임.

  ⚠ v13 의 G1·G3 통합(long-format)은 쓰지 않고 그룹 독립으로 4조건을 비교함(step18 과 동일).
     기준선도 같은 조건이므로 짝비교는 공정함. 통과하면 통합을 얹어 재측정함.

실행
    python scripts/step23_grid_dispersion.py --config configs/config_v13.yaml
    python scripts/step23_grid_dispersion.py --config configs/config_v13.yaml --topk 200,300
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
BAR = "=" * 84
CREF = 21600.0

# 격자 산포를 계산할 기저 변수 (§3.4 생존 계열 + 안정도 대리변수). 고정 목록 = 폴드 미참조
BASE = {
    "ldaps": ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
              "etc_0_blh", "heightAboveGround_2_t", "surface_0_NDNLW"],
    "gfs": ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
            "heightAboveGround_100_100u", "heightAboveGround_100_100v",
            "surface_0_gust"],
}
WSPAIR = {"ldaps": ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
          "gfs": ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100")}


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


def grid_meta(raw, prefix):
    """격자 id -> (위도, 경도). 경도 최소제곱 설계행렬을 만들기 위해 필요함."""
    g = raw.drop_duplicates("grid_id").sort_values("grid_id")
    return (g["grid_id"].to_numpy(), g["latitude"].to_numpy(float), g["longitude"].to_numpy(float))


def add_dispersion(df, metas, verbose=True):
    """
    격자 간 산포·공간 경도. 설계행렬이 시간에 대해 고정이므로 한 번의 행렬곱으로 끝남.
        v(grid) ≈ a + b·(lon − lon̄) + c·(lat − lat̄)     최소제곱
        b = 동서 경도(_gdx), c = 남북 경도(_gdy)
    """
    new, made = {}, 0
    for prefix, (gids, lat, lon) in metas.items():
        x = (lon - lon.mean()) * 111.32 * np.cos(np.radians(float(lat.mean())))   # km
        y = (lat - lat.mean()) * 110.57                                           # km
        A = np.vstack([np.ones(len(gids)), x, y]).T
        P = np.linalg.pinv(A)                                   # (3, n_grid) — 고정 연산자

        def stack(var):
            cols = [f"{prefix}_g{g}_{var}" for g in gids]
            cols = [c for c in cols if c in df.columns]
            if len(cols) < len(gids) * 0.8:
                return None
            return df[cols].to_numpy(float)                     # (n_row, n_grid)

        vars_ = list(BASE[prefix])
        mats = {v: stack(v) for v in vars_}
        # 풍속은 격자별로 합성해서 추가
        ua, va, wsname = WSPAIR[prefix]
        if mats.get(ua) is not None and mats.get(va) is not None:
            mats[wsname] = np.sqrt(mats[ua] ** 2 + mats[va] ** 2)
            vars_.append(wsname)

        for v, M in mats.items():
            if M is None:
                continue
            mu = np.nanmean(M, axis=1)
            sd = np.nanstd(M, axis=1, ddof=1)
            new[f"{prefix}_{v}_gstd"] = sd
            new[f"{prefix}_{v}_grange"] = np.nanmax(M, axis=1) - np.nanmin(M, axis=1)
            co = np.nan_to_num(M, nan=0.0) @ P.T                # (n_row, 3)
            new[f"{prefix}_{v}_gdx"] = co[:, 1]
            new[f"{prefix}_{v}_gdy"] = co[:, 2]
            if v in (wsname,):
                new[f"{prefix}_{v}_gcv"] = sd / (np.abs(mu) + 1e-6)
                new[f"{prefix}_{v}_ggrad"] = np.hypot(co[:, 1], co[:, 2])
                # H1 핵심 상호작용: 공간 산포 × 풍속 수준
                new[f"{prefix}_{v}_gstd_x_mean"] = sd * mu
            made += 1
    out = pd.DataFrame(new, index=df.index)
    if verbose:
        print(f"  격자 산포 피처 {out.shape[1]}개 (기저 {made}변수 × std/range/gdx/gdy [+cv/ggrad/상호작용])")
    return out


def run_arm(df, feats, targets, folds, fds, scale, mtype, mparams, esr, sd, top_k, newset):
    oof = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    picked = []
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
            picked.append(sum(1 for c in cols if c in newset))
            m, _ = fit_no_leak(mtype, mparams, df[cols].iloc[trf], df[g].iloc[trf], fds[trf],
                               es_rounds=esr, mode="refit", seed=sd)
            pred[va] = np.clip(m.predict(df[cols].iloc[va]), 0, 1.15 * CREF)
        oof[g] = np.clip(pred * scale[g], 0, CAPACITY_KWH[g])
    return oof, (float(np.mean(picked)) if picked else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--out-dir", default="./saved_models/_ab_step23")
    ap.add_argument("--topk", default="200,250")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed_everything(cfg["seed"])
    targets = cfg["targets"]; sd = cfg["seed"]
    esr = cfg.get("early_stopping_rounds", 30)
    mtype = cfg.get("model_type", "XGBoost"); mparams = cfg.get("model_params", {})
    K1, K2 = [int(v) for v in args.topk.split(",")]
    odir = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1  데이터 + 피처 + 격자 산포")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    rl = pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig")
    rg = pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig")
    metas = {"ldaps": grid_meta(rl, "ldaps"), "gfs": grid_meta(rg, "gfs")}
    print(f"  격자 수: LDAPS {len(metas['ldaps'][0])}  GFS {len(metas['gfs'][0])}")
    w = process_weather_data(rl, "ldaps").merge(process_weather_data(rg, "gfs"),
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

    disp = add_dispersion(df, metas)
    df = pd.concat([df, disp], axis=1).replace([np.inf, -np.inf], np.nan)

    excl = set(["forecast_kst_dtm", "kst_dtm", "data_available_kst_dtm", "year_month",
                "_year", "_fday"] + targets + [f"{g}_cf" for g in targets]
               + [f"{g}_avail_frac" for g in targets])
    allnum = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
    newset = set(disp.columns)
    base = [c for c in allnum if c not in newset]
    full = base + [c for c in allnum if c in newset]
    print(f"  기존 {len(base)}  +산포 {len(full)-len(base)}  =  {len(full)}")
    print("  예시: " + ", ".join(list(disp.columns)[:5]) + " ...")

    yrs, fds = df["_year"].to_numpy(), df["_fday"].to_numpy()
    folds = make_folds(yrs, fds, scheme="loyo")

    ARMS = {f"A 기준 K{K1}": (base, K1), f"B +산포 K{K1}": (full, K1),
            f"C 기준 K{K2}": (base, K2), f"D +산포 K{K2}": (full, K2)}

    print(BAR); print(f"STEP 2  4개 조건 학습 (그룹 독립, 나머지 v13 고정)")
    oofs, pick = {}, {}
    for name, (fe, k) in ARMS.items():
        oofs[name], pick[name] = run_arm(df, fe, targets, folds, fds, scale,
                                         mtype, mparams, esr, sd, k, newset)
        s = total_score(answer, oofs[name], targets)
        oofs[name].to_csv(odir / f"oof_{name.split()[0]}.csv")
        print(f"  {name:14s} raw {s[0]:.4f}  1-NMAE {s[1]:.4f}  FICR {s[2]:.4f}  "
              f"|  top-K 중 신규 {pick[name]:.0f}개  ({time.time()-t0:.0f}s)")

    print(BAR); print("STEP 3  주효과 분해 (2x2)")
    k = list(ARMS)
    A, B, C, D = [total_score(answer, oofs[x], targets)[0] for x in k]
    print(f"  격자산포 주효과  ((B−A)+(D−C))/2 = {((B-A)+(D-C))/2:+.4f}")
    print(f"  선택예산 주효과  ((C−A)+(D−B))/2 = {((C-A)+(D-B))/2:+.4f}   (K{K1} → K{K2})")
    print(f"  상호작용         (D−B−C+A)       = {D-B-C+A:+.4f}")
    print(f"  → 상호작용이 크게 양수면 '피처를 더할 때 선택 예산도 같이 늘려야 한다'(§3.18)의 직접 증거임")

    print(BAR); print("STEP 4  판정 — raw OOF 연도짝비교 (§3.6 규칙 1·2·6)")
    print("  [연도별 raw]")
    for n, o in oofs.items():
        score_by_year(df, answer, o, targets, label=n)
    for n in k[1:]:
        print(f"\n  [{n} vs {k[0]}]")
        res = is_difference_real(df, answer, oofs[n], oofs[k[0]], targets, name_a=n, name_b=k[0])
        gpos, line = 0, []
        for g in targets:
            cap = CAPACITY_KWH[g]; a = answer[g].to_numpy(float)
            ga = group_score(a, oofs[k[0]][g].to_numpy(float), cap)[0]
            gk = group_score(a, oofs[n][g].to_numpy(float), cap)[0]
            gpos += int(np.isfinite(gk - ga) and gk > ga)
            line.append(f"{g.replace('kpx_','')} {gk-ga:+.4f}")
        ok = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2)
        print("   그룹별: " + "  ".join(line) + f"   (양수 {gpos}/{len(targets)})")
        print(f"   => {'✅ 채택 후보' if ok else '❌ 채택 안 함'}")
        if ok and "G3" not in "":
            pass

    print(BAR); print(f"💾 {odir}/   ⏱ {time.time()-t0:.0f}초")
    print("사전 예측 (§3.20): 격자 산포가 유효하다면 **G3 에서 가장 큰 개선**이 나와야 함.")
    print("  G1·G2 에 더 크게 나오면 기전이 다른 것이므로 §3.20 해석을 다시 씀.")
    print("통과 시 다음: (1) step4_prune_sweep 을 신규 피처 포함 상태로 재실행해 최적 K 재확정")
    print("             (2) 통합(long-format) 얹어 재측정  (3) LB 1회")


if __name__ == "__main__":
    main()