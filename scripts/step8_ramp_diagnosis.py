"""
scripts/step8_ramp_diagnosis.py
==============================================================================
2023년은 왜 어려웠나 — 램프 가설 검증

배경 (강원지방기상청 연 기후특성 보도자료)
  2023: 1·11·12월 "따뜻한 이동성고기압 영향 후 시베리아 기압능 급발달,
        북극 찬 공기 유입" -> 기온변동폭 11월 6.4℃(역대1위), 12월 6.1℃(1위), 1월 4.5℃(10위)
  2022: 12월 대륙고기압 지배로 -4.2℃(하위4위). 안정된 겨울 패턴
  2024: 티베트·북태평양 고기압 발달 우세

가설
  기온변동이 크다 = 종관 패턴이 빠르게 교대한다 = 풍속 램프가 잦다.
  12~35시간 선행예보에서 램프 타이밍을 틀리면 오차가 커진다.
  게다가 11·12·1월은 발전량 최대 시기이고 FICR 은 발전량 가중이라 손실이 배가된다.

  => 2023년의 열세는 '겨울철 램프 다발' 로 설명될 것이다.

⚠️ 이 보도자료는 연말 이후 발간되므로 **피처로 쓸 수 없다**(규칙 3항).
   가설 생성용이며, 검증에 쓰는 램프 지표는 예보 데이터에서만 계산한다.

실행:
    python scripts/step8_ramp_diagnosis.py --config configs/config_v13.yaml \
        --oof saved_models/_ab_step5/oof_A.csv
==============================================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_full_feature_pipeline
from src.utils import CAPACITY_KWH
from src.validation import quiet_warnings, add_time_keys, group_score

quiet_warnings()
BAR = "=" * 78


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oof", default="./saved_models/_ab_step5/oof_A.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    tdir = Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1/5  데이터 + 예보 파생")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = add_time_keys(build_full_feature_pipeline(
        labels.merge(w, on="forecast_kst_dtm", how="left")).replace([np.inf, -np.inf], np.nan))
    answer = df[targets].copy()

    t = pd.to_datetime(df["forecast_kst_dtm"])
    df["_month"] = t.dt.month
    df["_date"] = t.dt.date

    # --- 램프 지표: 반드시 예보 블록 안에서만 계산 (규칙 3항) ---
    L = "ldaps_mean_"
    u10, v10 = df.get(L + "heightAboveGround_10_10u"), df.get(L + "heightAboveGround_10_10v")
    if u10 is None:
        print("❌ LDAPS 10m 컬럼을 찾을 수 없다."); return
    ws = np.sqrt(u10.to_numpy(float) ** 2 + v10.to_numpy(float) ** 2)
    df["_ws"] = ws
    gb = df.groupby(pd.to_datetime(df["data_available_kst_dtm"]), sort=False)["_ws"]
    df["_ramp1"] = gb.diff(1).abs()
    df["_blk_range"] = gb.transform("max") - gb.transform("min")

    tcol = L + "heightAboveGround_2_t"
    if tcol in df:
        temp = df[tcol].to_numpy(float)
        df["_t"] = np.where(temp > 100, temp - 273.15, temp)

    print(f"  행 {len(df):,}  |  램프 지표는 예보 블록(groupby data_available) 안에서만 계산")

    # --- OOF ---
    oof = pd.read_csv(args.oof, index_col=0).reindex(index=df.index, columns=targets)
    print(f"  OOF: {args.oof}")

    print(BAR); print("STEP 2/5  기온변동폭 — 보도자료와 대조")
    if "_t" in df:
        dm = df.groupby(["_year", "_month", "_date"])["_t"].mean().reset_index()
        var = dm.groupby(["_year", "_month"])["_t"].std().unstack()
        print("  [월별 일평균기온 표준편차 (℃)]  ※ 보도자료: 2023년 11월 6.4(1위), 12월 6.1(1위)")
        print("   " + var.round(2).to_string().replace("\n", "\n   "))
        w_ = [11, 12, 1]
        wv = var[[m for m in w_ if m in var.columns]].mean(axis=1)
        print("\n  겨울(11·12·1월) 평균 변동폭: " + ", ".join(f"{y}={v:.2f}" for y, v in wv.items()))
        print(f"   -> {'✅ 2023년이 최대 — 보도자료와 일치' if wv.idxmax()==2023 else '⚠️ 2023년이 최대가 아님'}")

    print(BAR); print("STEP 3/5  램프 강도 — 연도·월별")
    rp = df.groupby(["_year", "_month"])["_ramp1"].mean().unstack()
    print("  [시간당 |Δ풍속| 평균 (m/s)]")
    print("   " + rp.round(3).to_string().replace("\n", "\n   "))
    wr = rp[[m for m in [11, 12, 1] if m in rp.columns]].mean(axis=1)
    print("\n  겨울 평균 램프: " + ", ".join(f"{y}={v:.3f}" for y, v in wr.items()))
    print(f"   -> {'✅ 2023년 겨울 램프가 최대' if wr.idxmax()==2023 else '⚠️ 2023년이 최대가 아님'}")

    print(BAR); print("STEP 4/5  오차 분해 — 어느 달이 2023년을 끌어내렸나")
    rows = []
    for y in sorted(df["_year"].unique()):
        for m in range(1, 13):
            sel = (df["_year"] == y) & (df["_month"] == m)
            if sel.sum() < 200:
                continue
            ss = []
            for g in targets:
                mm = sel & answer[g].notna() & oof[g].notna()
                if mm.sum() < 200:
                    continue
                ss.append(group_score(answer.loc[mm, g].to_numpy(float),
                                      oof.loc[mm, g].to_numpy(float), CAPACITY_KWH[g])[0])
            if ss:
                rows.append({"year": y, "month": m, "score": np.mean(ss),
                             "ramp": df.loc[sel, "_ramp1"].mean(),
                             "gen": answer.loc[sel, targets[0]].mean()})
    M = pd.DataFrame(rows)
    piv = M.pivot(index="month", columns="year", values="score")
    print("  [월별 점수]")
    print("   " + piv.round(4).to_string().replace("\n", "\n   "))
    if 2023 in piv.columns:
        others = [c for c in piv.columns if c != 2023]
        gap = piv[2023] - piv[others].mean(axis=1)
        print("\n  [2023 − 타연도 평균]  (음수 = 2023이 나쁨)")
        print("   " + gap.round(4).to_string().replace("\n", "\n   "))
        worst = gap.nsmallest(3)
        print(f"\n  2023 이 가장 뒤처진 달: " + ", ".join(f"{m}월({v:+.4f})" for m, v in worst.items()))
        win = [m for m in [11, 12, 1] if m in gap.index]
        print(f"  겨울(11·12·1월) 평균 격차: {gap[win].mean():+.4f}  "
              f"/ 그 외 {gap[[m for m in gap.index if m not in win]].mean():+.4f}")
        print(f"   -> {'✅ 겨울이 유독 나쁨 — 가설 지지' if gap[win].mean() < gap[[m for m in gap.index if m not in win]].mean() - 0.005 else '⚠️ 겨울에 특별히 몰리지 않음'}")

    print(BAR); print("STEP 5/5  램프 강도와 오차의 직접 관계")
    ok = df["_ramp1"].notna()
    q = pd.qcut(df.loc[ok, "_ramp1"], 5, labels=["최소", "낮음", "중간", "높음", "최대"])
    rows = []
    for lab in q.cat.categories:
        sel = ok.copy(); sel[ok] = (q == lab).to_numpy()
        ss, ns = [], 0
        for g in targets:
            mm = sel & answer[g].notna() & oof[g].notna()
            if mm.sum() < 200:
                continue
            ss.append(group_score(answer.loc[mm, g].to_numpy(float),
                                  oof.loc[mm, g].to_numpy(float), CAPACITY_KWH[g])[0])
            ns = int(mm.sum())
        rows.append({"램프": lab, "점수": np.mean(ss), "평균|Δv|": df.loc[sel, "_ramp1"].mean(),
                     "평균발전량": answer.loc[sel, targets[0]].mean(), "n": ns})
    Q = pd.DataFrame(rows)
    print("   " + Q.round(4).to_string(index=False).replace("\n", "\n   "))
    d = Q["점수"].iloc[0] - Q["점수"].iloc[-1]
    print(f"\n  최소 램프 − 최대 램프 = {d:+.4f}")
    print(f"   -> {'✅ 램프가 클수록 확실히 나쁘다' if d > 0.02 else '⚠️ 램프 효과가 뚜렷하지 않다'}")
    print(BAR)


if __name__ == "__main__":
    main()