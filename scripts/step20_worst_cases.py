"""
scripts/step20_worst_cases.py
==============================================================================
[도구] 잔차 케이스 분석 — 집계가 아니라 '무슨 일이 있었나' 를 본다

왜 필요한가
  v13 이후 5연속 기각(step12/13/14/16/18)이다. 전부 '가설 -> 집계통계 -> 판정' 이었고,
  EDA 이후로 **잔차로 고른 실제 행을 열어본 적이 한 번도 없다.**
  집계는 '어떤 가설이 맞나' 는 답하지만 '어떤 가설을 세울까' 는 답하지 못한다.
  기각이 연속되면 가설 생성원이 마른 것이므로 데이터로 돌아간다.

핵심 질문 — 최악 케이스는 어느 범주인가. 범주가 다음 행동을 가른다.
    가용률 저하    -> 정제 타깃의 min_avail_frac 임계 재검토 (코드 한 줄)
    램프          -> 시간 구조. A-6 이 죽었으니 다른 형태 필요
    순수 예보오차   -> 우리가 못 고친다. A-1(외부 NWP)만이 답. 나머지 트랙 재평가
    풍향/계절 편중  -> 후류·지형. A-4 / A-5
    컷인 무릎(L1)  -> §3.14. step17 이 이미 겨냥 중

'예보오차' 를 어떻게 판별하나 — 파워커브 역산
  실제 발전량을 SCADA 경험 파워커브로 역산하면 **그 시간에 실제로 불었을 풍속**이 나온다.
  그것을 예보 풍속과 비교하면 오차의 출처가 갈린다.
      |역산풍속 − 예보풍속| 이 크다  -> 예보가 틀렸다
      작은데도 발전량이 안 나왔다     -> 터빈 쪽 문제(가용률/제어)
  정격(약 11 m/s) 위에서는 곡선이 평평해 역산이 불가능하므로 cf < 0.85 에서만 쓴다.

학습 없음. 저장된 OOF 와 SCADA 파생물만 읽는다.

실행
    python scripts/step20_worst_cases.py --config configs/config_v13.yaml
    python scripts/step20_worst_cases.py --config configs/config_v13.yaml --n 150 --show-days 5
    python scripts/step20_worst_cases.py --config configs/config_v13.yaml --group kpx_group_3
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
from src.utils import CAPACITY_KWH
from src.validation import quiet_warnings, add_time_keys, total_score
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 100
MAKER = {"kpx_group_1": "vestas", "kpx_group_2": "vestas", "kpx_group_3": "unison"}
NTURB = {"kpx_group_1": 6, "kpx_group_2": 6, "kpx_group_3": 5}

# PART 2 에서 최악 vs 전체를 비교할 변수 (없으면 자동 스킵)
CTX = ["gfs_v117_powerlaw", "ldaps_mean_heightAboveGround_10_10u",
       "ldaps_mean_heightAboveGround_10_10v", "wpd_ldaps_10m", "wpd_gfs_117m",
       "gfs_alpha_shear", "gfs_gust_ratio_100m", "ldaps_mean_etc_0_blh",
       "ldaps_blh_v117_interaction", "ldaps_ws50_turb_range", "ldaps_sw_total",
       "ldaps_mean_heightAboveGround_2_t", "gfs_lapse_rate_850",
       "gfs_v850_inflow_ratio", "wind_dir_deg", "wind_perp_ridge",
       "abs_diff_ws10_ldaps_gfs", "gfs_v117_ramp_1h", "hour", "month"]


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


def invert_power_curve(kwh_per_turbine, maker, scada_dir):
    """
    실제 발전량 -> '그 시간에 실제로 불었을 풍속'. 곡선의 단조 증가 구간만 쓴다.
    정격 위에서는 곡선이 평평해 역산이 불가능하므로 NaN 을 돌려준다.
    """
    p = Path(scada_dir) / f"power_curve_{maker}.csv"
    if not p.exists():
        return np.full(len(kwh_per_turbine), np.nan)
    d = pd.read_csv(p, encoding="utf-8-sig")
    ws, kwh = d["ws"].to_numpy(float), d["kwh_per_h"].to_numpy(float)
    rated = np.nanmax(kwh)
    keep = (ws <= 24) & (kwh < rated * 0.98)                  # 정격 도달 전까지
    if keep.sum() < 5:
        return np.full(len(kwh_per_turbine), np.nan)
    ws_m, kwh_m = ws[keep], np.maximum.accumulate(kwh[keep])
    out = np.interp(kwh_per_turbine, kwh_m, ws_m, left=np.nan, right=np.nan)
    out[np.asarray(kwh_per_turbine, float) > rated * 0.85] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-base", default="./saved_models/v13")
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--n", type=int, default=120, help="살펴볼 최악 케이스 수 (전 그룹 합)")
    ap.add_argument("--show-rows", type=int, default=25, help="표로 찍을 행 수")
    ap.add_argument("--show-days", type=int, default=3, help="24시간 시계열로 펼칠 최악 블록 수")
    ap.add_argument("--group", default=None, help="한 그룹만 보기")
    ap.add_argument("--out-csv", default="./saved_models/_ab_step20/worst_cases.csv")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = [args.group] if args.group else cfg["targets"]
    db, tdir = Path(args.dir_base), Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1  데이터 + 피처 + OOF + 후처리")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = add_time_keys(build_full_feature_pipeline(
        labels.merge(w, on="forecast_kst_dtm", how="left")).replace([np.inf, -np.inf], np.nan))
    answer = df[cfg["targets"]].copy()
    t = pd.to_datetime(df["forecast_kst_dtm"])
    yrs = df["_year"].to_numpy()

    # 가동률 (SCADA 파생) — 있으면 붙인다
    avail = {}
    ap_path = Path(args.scada_dir) / "available_capacity.csv"
    if ap_path.exists():
        ac = pd.read_csv(ap_path, encoding="utf-8-sig")
        ac["forecast_kst_dtm"] = pd.to_datetime(ac.pop("kst_dtm"))
        ac = df[["forecast_kst_dtm"]].merge(ac, on="forecast_kst_dtm", how="left")
        for g in cfg["targets"]:
            if g in ac:
                avail[g] = ac[g].to_numpy(float) / CAPACITY_KWH[g]
        print(f"  가동률 결합: {ap_path}")
    else:
        print(f"  ⚠ {ap_path} 없음 — 가용률 범주는 판정 불가")

    ob = pd.read_csv(db / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=cfg["targets"]).astype(float)
    post = pd.DataFrame(index=df.index, columns=cfg["targets"], dtype=float)
    for y in [int(v) for v in sorted(pd.unique(yrs)) if (yrs == v).sum() >= 200]:
        te = (yrs == y); tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        post.loc[te, cfg["targets"]] = apply_postprocessing(ob.loc[te].copy(), pp)[cfg["targets"]].values
    s0 = total_score(answer, ob, cfg["targets"])
    print(f"  기준 raw OOF {s0[0]:.4f}   행 {len(df):,}")

    # ---------------------------------------------------------------- 케이스 표 만들기
    rows = []
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float)
        p = ob[g].to_numpy(float)
        q = post[g].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(q) & (a >= 0.10 * cap)
        err = np.abs(q - a) / cap * 100                              # 후처리 후 오차 %cap
        # 앞뒤 시간 실제값 (같은 그룹, 시각 기준. 램프 판정용)
        a_prev = pd.Series(a).shift(1).to_numpy()
        a_next = pd.Series(a).shift(-1).to_numpy()
        implied = invert_power_curve(a / NTURB[g], MAKER[g], args.scada_dir)
        d = pd.DataFrame({
            "group": g, "dtm": t, "hour": t.dt.hour, "month": t.dt.month, "year": yrs,
            "actual%": a / cap * 100, "raw%": p / cap * 100, "post%": q / cap * 100,
            "err%": err, "bias%": (q - a) / cap * 100,
            "ramp_prev%": (a - a_prev) / cap * 100, "ramp_next%": (a_next - a) / cap * 100,
            "avail": avail.get(g, np.full(len(df), np.nan)),
            "ws_implied": implied,
            "v117_fc": df.get("gfs_v117_powerlaw", pd.Series(np.nan, index=df.index)).to_numpy(float),
            "wdir": df.get("wind_dir_deg", pd.Series(np.nan, index=df.index)).to_numpy(float),
            "blh": df.get("ldaps_mean_etc_0_blh", pd.Series(np.nan, index=df.index)).to_numpy(float),
            "_ok": ok,
        })
        d["ws_gap"] = d["ws_implied"] - d["v117_fc"]
        rows.append(d)
    C = pd.concat(rows, ignore_index=True)
    scored = C[C["_ok"]].copy()
    worst = scored.nlargest(args.n, "err%").copy()

    # ================================================================ PART 1
    print(BAR)
    print(f"PART 1  최악 {args.n}개 중 상위 {args.show_rows}개  (오차는 후처리 후 %cap. 밴드 경계 6%)")
    print("  ws_implied = 실제 발전량을 파워커브로 역산한 풍속 / v117_fc = 예보 풍속 / gap = 역산−예보\n")
    cols = ["dtm", "group", "hour", "actual%", "post%", "err%", "bias%",
            "ramp_prev%", "ramp_next%", "avail", "ws_implied", "v117_fc", "ws_gap", "wdir"]
    show = worst.head(args.show_rows)[cols].copy()
    show["group"] = show["group"].str.replace("kpx_group_", "G")
    show["dtm"] = show["dtm"].dt.strftime("%Y-%m-%d %H:%M")
    print(show.to_string(index=False, float_format=lambda v: f"{v:7.1f}"))

    # ================================================================ PART 2
    print(BAR)
    print("PART 2  최악 케이스는 무엇이 다른가 — 표준화 차이 (최악평균 − 전체평균) / 전체표준편차")
    print("  |z| > 0.5 면 눈에 띄는 차이다. 부호가 방향을 알려준다.\n")
    ctx = [c for c in CTX if c in df.columns]
    base = df.loc[scored.index.map(lambda i: i % len(df)) if False else slice(None)]
    diffs = []
    for c in ctx:
        v_all = df[c].to_numpy(float)
        # 최악 케이스의 원본 행 인덱스 복원 (C 는 그룹별로 df 를 반복한 구조)
        idx = worst.index.to_numpy() % len(df)
        v_w = v_all[idx]
        m_all, s_all = np.nanmean(v_all), np.nanstd(v_all)
        if not np.isfinite(s_all) or s_all == 0:
            continue
        diffs.append((c, (np.nanmean(v_w) - m_all) / s_all, np.nanmean(v_w), m_all))
    diffs.sort(key=lambda x: -abs(x[1]))
    print(f"  {'변수':<40s}{'z':>8s}{'최악평균':>12s}{'전체평균':>12s}")
    for c, z, mw, ma in diffs[:14]:
        flag = "   <-" if abs(z) > 0.5 else ""
        print(f"  {c[:40]:<40s}{z:+8.2f}{mw:12.2f}{ma:12.2f}{flag}")

    print(f"\n  [시각·계절 편중]")
    for k, lab in [("hour", "시각"), ("month", "월")]:
        vc = worst[k].value_counts().sort_index()
        exp = len(worst) / (24 if k == "hour" else 12)
        hot = [f"{int(i)}({int(n)})" for i, n in vc.items() if n >= exp * 1.8]
        print(f"    {lab} 과대표집(기대 {exp:.1f}회 대비 1.8배 이상): " + (", ".join(hot) or "없음"))
    print(f"    그룹 분포: " + "  ".join(
        f"{g.replace('kpx_group_','G')} {int((worst['group']==g).sum())}" for g in targets))

    # ================================================================ PART 3
    print(BAR)
    print("PART 3  자동 범주화 — 이 비율이 다음 행동을 정한다")
    print("  (우선순위대로 배타 분류. 위에서 걸리면 아래는 안 본다)\n")

    def classify(r):
        if np.isfinite(r["avail"]) and r["avail"] < 0.90:
            return "① 가용률 저하"
        if np.isfinite(r["ws_gap"]) and abs(r["ws_gap"]) >= 2.0:
            return "② 예보 풍속 오차"
        rp, rn = r["ramp_prev%"], r["ramp_next%"]
        if (np.isfinite(rp) and abs(rp) >= 15) or (np.isfinite(rn) and abs(rn) >= 15):
            return "③ 램프"
        if r["actual%"] < 20 and r["raw%"] < 15:
            return "④ 컷인 무릎 (L1)"
        if np.isfinite(r["ws_gap"]) and abs(r["ws_gap"]) < 2.0:
            return "⑤ 예보는 맞았는데 발전량이 안 맞음"
        return "⑥ 미분류"

    worst["cat"] = worst.apply(classify, axis=1)
    vc = worst["cat"].value_counts()
    for k in ["① 가용률 저하", "② 예보 풍속 오차", "③ 램프", "④ 컷인 무릎 (L1)",
              "⑤ 예보는 맞았는데 발전량이 안 맞음", "⑥ 미분류"]:
        n = int(vc.get(k, 0))
        bar = "█" * int(round(n / max(len(worst), 1) * 50))
        sub = worst[worst["cat"] == k]
        extra = f"  평균오차 {sub['err%'].mean():.1f}%cap" if n else ""
        print(f"  {k:<32s}{n:4d}  {n/len(worst)*100:5.1f}%  {bar}{extra}")

    print("\n  판정 가이드")
    print("    ① 가장 큼  -> clean_target 의 min_avail_frac(현재 0.5) 을 0.8~0.9 로 올려 재학습. 코드 한 줄")
    print("    ② 가장 큼  -> 순수 예보 오차다. A-3/A-4/A-5 로는 못 고친다. A-1(외부 NWP) 아니면 목표 재조정")
    print("    ③ 가장 큼  -> 시간 구조. A-6 은 죽었으니 다른 형태(예: 램프 전용 모델/가중) 필요")
    print("    ④ 가장 큼  -> §3.14 의 L1. step17(B-2)이 이미 겨냥 중. 그쪽에 집중")
    print("    ⑤ 가장 큼  -> 예보는 맞는데 발전량이 안 나온다. 후류/제어/미포착 가용률. A-4 또는 SCADA 재검토")

    # 대조군: 전체 채점행을 같은 규칙으로 분류해 '최악에서만 과대표집' 되는지 본다
    samp = scored.sample(n=min(4000, len(scored)), random_state=0).copy()
    samp["cat"] = samp.apply(classify, axis=1)
    sv = samp["cat"].value_counts(normalize=True) * 100
    print("\n  [대조군] 전체 채점행 4,000개 무작위 표본의 같은 분류 (최악과 비교해야 의미가 있다)")
    for k in vc.index:
        w_pct = vc[k] / len(worst) * 100
        a_pct = float(sv.get(k, 0.0))
        lift = w_pct / a_pct if a_pct > 0.1 else np.inf
        print(f"    {k:<32s} 최악 {w_pct:5.1f}%  전체 {a_pct:5.1f}%   lift {lift:5.1f}x")

    # ================================================================ PART 4
    print(BAR)
    print(f"PART 4  최악 블록 {args.show_days}개의 24시간 전개 — 하루가 통째로 나쁜지, 한 시간만 튀는지")
    worst["day"] = worst["dtm"].dt.normalize()
    top_days = worst.groupby(["group", "day"])["err%"].agg(["mean", "count"]) \
                    .sort_values("mean", ascending=False)
    top_days = top_days[top_days["count"] >= 2].head(args.show_days)
    for (g, day), r in top_days.iterrows():
        sub = C[(C["group"] == g) & (C["dtm"].dt.normalize() == day)].sort_values("dtm")
        print(f"\n  [{g.replace('kpx_group_','G')}] {day.date()}  "
              f"최악시간 {int(r['count'])}개, 평균오차 {r['mean']:.1f}%cap")
        print("    " + f"{'시':>3s}{'실제%':>8s}{'예측%':>8s}{'오차%':>8s}{'예보v117':>10s}{'역산ws':>9s}{'가용률':>8s}")
        for _, x in sub.iterrows():
            mark = " ***" if x["err%"] >= 10 and x["_ok"] else ("  --" if not x["_ok"] else "")
            print("    " + f"{int(x['hour']):3d}{x['actual%']:8.1f}{x['post%']:8.1f}"
                  + (f"{x['err%']:8.1f}" if x["_ok"] else f"{'-':>8s}")
                  + f"{x['v117_fc']:10.1f}"
                  + (f"{x['ws_implied']:9.1f}" if np.isfinite(x["ws_implied"]) else f"{'-':>9s}")
                  + (f"{x['avail']:8.2f}" if np.isfinite(x["avail"]) else f"{'-':>8s}") + mark)
        print("    (-- 는 채점 대상 아님(실제 <10%cap), *** 는 오차 10%cap 이상)")

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    worst.drop(columns=["_ok"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(BAR); print(f"💾 {out}   ⏱ {time.time()-t0:.0f}초")
    print("이 표를 보고 '무엇을 재볼지' 를 정한다. 여기서 나온 가설만 실험으로 올린다.")


if __name__ == "__main__":
    main()