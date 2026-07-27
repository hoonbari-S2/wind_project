"""
scripts/step21_highwind.py
==============================================================================
고풍속 컷아웃 — §3.14 가 닫은 'p 의 함수 공간' 을 뚫는 유일한 예외

step20 이 찾은 것
  최악 120개 중 115개가 **과대예측**이고, 단일 패턴이다.
      wpd z=+2.31 / v117 z=+1.74 / u10 z=+1.44 / wdir 250~275°에 집중
      => "예보는 강한 서풍인데 실제 발전량이 10~28%cap"
  모델은 그 시간에 85~92%cap 을 예측했고 후처리가 100%로 더 밀어올렸다.

원인 두 갈래
  (a) 가용률 저하 — 2024-01-22~24 G2 가 3일 연속 1/6대. 추론 시점에 알 수 없다. 못 고친다.
  (b) 고풍속 컷아웃/스톰 제어 — **avail=1.0 인데 발전량이 없는 시간이 있다.**
      §4.3 이 stopped_weather 를 의도적으로 '가용' 으로 유지했다(예측 가능한 현상이므로).
      설계는 맞는데 모델이 못 배우고 있다.

왜 못 배우나 — 임계값이 전부 가정이다
      turbine_active_flag = (v>=3.5) & (v<25.0)   # 예보 최대가 21이라 항상 1. 무용
      soft_cutout_factor : 20 m/s 부터 감쇠        # 21 m/s 에서 0.82. 너무 약함
      scada.py  ref[centers > 26] = 0.0            # 26 은 측정값이 아니라 가정
      scada.py  ref[centers<=24] = 최대누적          # 24 m/s 까지 단조 강제
                => **저장된 경험 파워커브는 구성상 컷아웃을 표현할 수 없다.**

왜 이것이 §3.14 의 예외인가
  후처리는 np.maximum.accumulate 로 단조를 강제한다. 컷아웃은 비단조다.
  단조 변환으로는 원리적으로 표현할 수 없다.
  단조를 풀어도 안 된다 — p 가 높은 이유가 (i) 정상 고출력 (ii) 컷아웃 둘인데
  p 만으로는 구분이 안 되므로 정상 고출력까지 같이 깎인다.
  **구분하려면 예보 풍속 축이 필요하고, 그것은 p 의 함수가 아니다.**
  §3.14 가 닫은 것은 'p 의 함수 공간' 이었고 step14 가 본 것은 시각·계절 축이었다.
  **풍속 축은 한 번도 시험한 적이 없다.**
  그리고 이 구간에서 p 는 무정보가 아니라 **서로 다른 두 상태를 같은 값으로 뭉개고 있다.**

--------------------------------------------------------------------------
PART 0  못 고치는 몫을 먼저 확정한다 — 가용률 구간별 점수 손실
PART 1  진단 (avail == 1.0 행만) — 예보 풍속 구간별 편향·밴드적중률
PART 2  저장된 파워커브가 컷아웃을 담고 있는지 확인
PART 3  적합 — v117 > TH 인 행에 배율 m. (TH=∞ 또는 m=1 이면 현행과 동일)

판정 (사전 등록, §3.6 규칙 6)
  부호 3/3 일치 ∧ **부호 양수** ∧ |평균| > 표준편차 ∧ 그룹 2개 이상 양수. 넷 다여야 제출.
  추가 전제: PART 1 에서 avail=1.0 고풍속 구간의 평균 편향이 +5%cap 이상이어야 한다.
            그보다 작으면 기전이 없는 것이므로 PART 3 결과가 통과해도 제출하지 않는다.

학습 없음. 저장된 OOF 와 SCADA 파생물만 읽는다.

실행
    python scripts/step21_highwind.py --config configs/config_v13.yaml
    python scripts/step21_highwind.py --config configs/config_v13.yaml --wind ldaps_ws10
    python scripts/step21_highwind.py --config configs/config_v13.yaml --make-submission
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
from src.validation import (quiet_warnings, add_time_keys, total_score, group_score,
                            is_difference_real)
from src.postprocessing import optimize_postprocessing, apply_postprocessing

quiet_warnings()
BAR = "=" * 92
MAKER = {"kpx_group_1": "vestas", "kpx_group_2": "vestas", "kpx_group_3": "unison"}
WS_EDGES = [0, 5, 8, 11, 14, 17, 20, 99]
AV_EDGES = [0.0, 0.5, 0.8, 0.999, 1.001]
AV_NAME = ["<0.5", "0.5~0.8", "0.8~1.0", "= 1.0"]
TH_GRID = [11.0, 13.0, 15.0, 17.0, 19.0, 999.0]     # 999 = 적용 안 함(= 현행)
M_GRID = [0.30, 0.45, 0.60, 0.75, 0.90, 1.00]       # 1.00 = 현행
MIN_CELL = 60


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


def band_hit(a, f, cap):
    """FICR 밴드 적중률 (발전량 가중, ≤6% 기준). 점수와 직결되는 지표."""
    e = np.abs(f - a) / cap
    if len(a) == 0:
        return np.nan
    return float((a * (e <= 0.06)).sum() / max(a.sum(), 1e-9) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir-base", default="./saved_models/v13")
    ap.add_argument("--scada-dir", default="./data/scada_derived")
    ap.add_argument("--wind", default="gfs_v117_powerlaw",
                    help="풍속 축. gfs_v117_powerlaw | ldaps_ws10 (스크립트가 계산)")
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--out-name", default="submit_v15_highwind.csv")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    targets = cfg["targets"]
    db, tdir = Path(args.dir_base), Path(cfg["data_paths"]["train_dir"])

    print(BAR); print("STEP 1  데이터 + OOF + LOYO 중첩 후처리")
    labels = pd.read_csv(tdir / "train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels.pop("kst_dtm"))
    w = process_weather_data(pd.read_csv(tdir / "ldaps_train.csv", encoding="utf-8-sig"), "ldaps").merge(
        process_weather_data(pd.read_csv(tdir / "gfs_train.csv", encoding="utf-8-sig"), "gfs"),
        on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    w = w.drop(columns=[c for c in w.columns if c.endswith("_dup")])
    df = add_time_keys(build_full_feature_pipeline(
        labels.merge(w, on="forecast_kst_dtm", how="left")).replace([np.inf, -np.inf], np.nan))
    answer = df[targets].copy()
    yrs = df["_year"].to_numpy()
    years = [int(y) for y in sorted(pd.unique(yrs)) if (yrs == y).sum() >= 200]

    # 풍속 축
    if args.wind == "ldaps_ws10":
        u = df["ldaps_mean_heightAboveGround_10_10u"].to_numpy(float)
        v = df["ldaps_mean_heightAboveGround_10_10v"].to_numpy(float)
        wind = np.sqrt(u * u + v * v)
        wname = "ldaps_ws10 (10m, §3.4 스피어만 0.762)"
    else:
        wind = df[args.wind].to_numpy(float)
        wname = f"{args.wind} (§3.4 스피어만 0.689)"

    # 가용률
    avail = {}
    apth = Path(args.scada_dir) / "available_capacity.csv"
    if apth.exists():
        ac = pd.read_csv(apth, encoding="utf-8-sig")
        ac["forecast_kst_dtm"] = pd.to_datetime(ac.pop("kst_dtm"))
        ac = df[["forecast_kst_dtm"]].merge(ac, on="forecast_kst_dtm", how="left")
        for g in targets:
            if g in ac:
                avail[g] = ac[g].to_numpy(float) / CAPACITY_KWH[g]
    if not avail:
        print("  ⚠ available_capacity.csv 없음 — PART 0 과 PART 1 의 avail 필터가 무력화된다")

    ob = pd.read_csv(db / "oof_preds.csv", index_col=0).reindex(
        index=df.index, columns=targets).astype(float)
    post = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    pp_year = {}
    for y in years:
        te = (yrs == y); tr = ~te & answer.notna().any(axis=1).to_numpy()
        pp = optimize_postprocessing(answer.loc[tr], ob.loc[tr], mode="piecewise", verbose=False)
        pp_year[y] = pp
        post.loc[te, targets] = apply_postprocessing(ob.loc[te].copy(), pp)[targets].values
    s0 = total_score(answer, post, targets)
    print(f"  풍속 축: {wname}")
    print(f"  현행 중첩 후처리 총점 {s0[0]:.4f}   연도 {years}")

    # ================================================================ PART 0
    print(BAR)
    print("PART 0  못 고치는 몫 — 가용률 구간별 점수 손실 (추론 시점에 가용률은 알 수 없다)")
    print("  '완벽예측 시 이득' = 그 칸을 정답으로 바꿨을 때 총 NMAE 항이 얼마나 좋아지는가 (상한)\n")
    if avail:
        print("  " + f"{'가용률':<10s}{'채점행':>9s}{'비중':>8s}{'평균편향':>10s}{'밴드적중':>10s}"
              f"{'칸 점수':>10s}{'NMAE 손실':>11s}")
        tot_rows, tot_loss = 0, 0.0
        for k in range(len(AV_NAME)):
            n_all, bias_l, hit_l, sc_l, loss = 0, [], [], [], 0.0
            for g in targets:
                cap = CAPACITY_KWH[g]
                a = answer[g].to_numpy(float); q = post[g].to_numpy(float)
                av = avail.get(g)
                if av is None:
                    continue
                m = (np.isfinite(a) & np.isfinite(q) & (a >= 0.10 * cap)
                     & (av >= AV_EDGES[k]) & (av < AV_EDGES[k + 1]))
                if m.sum() < MIN_CELL:
                    continue
                n_all += int(m.sum())
                bias_l.append(np.mean(q[m] - a[m]) / cap * 100)
                hit_l.append(band_hit(a[m], q[m], cap))
                sc_l.append(group_score(a[m], q[m], cap)[0])
                loss += np.sum(np.abs(q[m] - a[m])) / cap        # NMAE 분자 기여
            if n_all == 0:
                continue
            tot_rows += n_all; tot_loss += loss
            print("  " + f"{AV_NAME[k]:<10s}{n_all:9d}{'':>8s}{np.mean(bias_l):+10.2f}"
                  f"{np.mean(hit_l):9.1f}%{np.mean(sc_l):10.4f}{loss:11.1f}")
        print(f"\n  전체 채점행 {tot_rows:,}  총 NMAE 분자 {tot_loss:.1f}")
        for k in range(len(AV_NAME) - 1):                          # avail<1.0 구간만
            pass
        print("  → 'avail = 1.0' 이 아닌 칸의 NMAE 손실 몫이 곧 **추론으로 못 고치는 상한**이다.")
        print("    그 몫이 크면 목표를 재조정하고, 작으면 아래 (b) 컷아웃에 집중한다.")
    else:
        print("  (가용률 파일이 없어 생략)")

    # ================================================================ PART 1
    print(BAR)
    print("PART 1  진단 — **가용률 1.0 행만**. 가용률 효과를 완전히 제거한 순수 예보 축 분석")
    print("  기전이 있다면 고풍속 구간에서 평균편향이 크게 양수(과대예측)여야 한다.\n")
    hi_bias = []
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float); q = post[g].to_numpy(float)
        av = avail.get(g, np.ones(len(df)))
        base = np.isfinite(a) & np.isfinite(q) & np.isfinite(wind) & (a >= 0.10 * cap) & (av >= 0.999)
        print(f"  [{g.replace('kpx_','')}]  가용률 1.0 채점행 {int(base.sum()):,}개")
        print("    " + f"{'예보풍속':<12s}{'행수':>7s}{'평균실제%':>10s}{'평균예측%':>10s}"
              f"{'편향%':>9s}{'밴드적중':>10s}{'칸점수':>9s}")
        for i in range(len(WS_EDGES) - 1):
            m = base & (wind >= WS_EDGES[i]) & (wind < WS_EDGES[i + 1])
            lab = f"{WS_EDGES[i]}~{WS_EDGES[i+1] if WS_EDGES[i+1]<99 else ''}"
            if m.sum() < MIN_CELL:
                print("    " + f"{lab:<12s}{int(m.sum()):7d}   (표본 부족)")
                continue
            bias = np.mean(q[m] - a[m]) / cap * 100
            print("    " + f"{lab:<12s}{int(m.sum()):7d}{np.mean(a[m])/cap*100:10.1f}"
                  f"{np.mean(q[m])/cap*100:10.1f}{bias:+9.2f}{band_hit(a[m],q[m],cap):9.1f}%"
                  f"{group_score(a[m],q[m],cap)[0]:9.4f}")
            if WS_EDGES[i] >= 14:
                hi_bias.append((bias, int(m.sum())))
        print()
    if hi_bias:
        wsum = sum(n for _, n in hi_bias)
        hb = sum(b * n for b, n in hi_bias) / max(wsum, 1)
        print(f"  고풍속(≥14 m/s) 가중 평균 편향 = {hb:+.2f}%cap  (행 {wsum:,})")
        gate = hb >= 5.0
        print(f"  => {'✅ 기전 있음. PART 3 로 간다.' if gate else '❌ 기전 약함. PART 3 가 통과해도 제출하지 않는다.'}")
    else:
        gate = False
        print("  ❌ 고풍속 표본이 부족하다. 이 축은 판정 불가.")

    # ================================================================ PART 2
    print(BAR)
    print("PART 2  저장된 경험 파워커브가 컷아웃을 담고 있는가")
    for mk in ["vestas", "unison"]:
        p = Path(args.scada_dir) / f"power_curve_{mk}.csv"
        if not p.exists():
            print(f"  {mk}: 파일 없음"); continue
        d = pd.read_csv(p, encoding="utf-8-sig")
        ws, kw = d["ws"].to_numpy(float), d["kwh_per_h"].to_numpy(float)
        rated = np.nanmax(kw)
        i_r = int(np.argmax(kw >= rated * 0.98)) if (kw >= rated * 0.98).any() else -1
        drop = ws[(ws > ws[i_r]) & (kw < rated * 0.5)] if i_r >= 0 else np.array([])
        print(f"  {mk}: 정격도달 {ws[i_r]:.1f} m/s  최대 {rated:.0f} kWh/h  "
              f"곡선 최대풍속 {ws.max():.1f} m/s  "
              + (f"50%로 떨어지는 지점 {drop.min():.1f} m/s" if len(drop) else "**하강 구간 없음**"))
    print("  ⚠ scada.py 의 fit_reference_curve 는 `ref[centers<=24]=maximum.accumulate` 로")
    print("    24 m/s 까지 단조를 강제하고 `ref[centers>26]=0` 으로 26 을 못박는다.")
    print("    => **이 곡선은 구성상 컷아웃을 담을 수 없다.** 위 '하강 구간 없음' 은 그 결과이지 측정이 아니다.")
    print("    실제 컷아웃 지점은 PART 1 의 '평균실제%' 가 꺾이는 곳에서 읽는다.")

    # ================================================================ PART 3
    print(BAR)
    print("PART 3  적합 — v117 > TH 인 행에 배율 m 을 곱한다.  (TH=999 또는 m=1.0 이면 현행)")
    print("  적합셋(그 해를 뺀 2년)에서만 TH·m 을 고르고 평가연도에 적용한다.\n")
    cur = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    new = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    for y in years:
        te = (yrs == y); tr = ~te & answer.notna().any(axis=1).to_numpy()
        post_tr = apply_postprocessing(ob.loc[tr].copy(), pp_year[y])
        post_te = apply_postprocessing(ob.loc[te].copy(), pp_year[y])
        print(f"  [{y}]")
        for g in targets:
            cap = CAPACITY_KWH[g]
            a_tr = answer.loc[tr, g].to_numpy(float)
            p_tr = post_tr[g].to_numpy(float); w_tr = wind[tr]
            ok = np.isfinite(a_tr) & np.isfinite(p_tr) & np.isfinite(w_tr)
            best, bth, bm = -np.inf, 999.0, 1.0
            for th in TH_GRID:
                hi = w_tr[ok] > th
                if hi.sum() < MIN_CELL and th < 999:
                    continue
                for m in M_GRID:
                    f = p_tr[ok].copy()
                    f[hi] = f[hi] * m
                    s = group_score(a_tr[ok], np.clip(f, 0, cap), cap)[0]
                    if np.isfinite(s) and s > best:
                        best, bth, bm = s, th, m
            p_te = post_te[g].to_numpy(float); w_te = wind[te]
            f_te = p_te.copy()
            hi_te = np.isfinite(w_te) & (w_te > bth)
            f_te[hi_te] = f_te[hi_te] * bm
            cur.loc[te, g] = p_te
            new.loc[te, g] = np.clip(f_te, 0, cap)
            a_te = answer.loc[te, g].to_numpy(float)
            gc = group_score(a_te, p_te, cap)[0]
            gn = group_score(a_te, new.loc[te, g].to_numpy(float), cap)[0]
            flat = "   <- 현행과 동일: 가설 무효" if (bm == 1.0 or bth >= 999) else ""
            print(f"    {g:14s} TH {bth:5.1f} / m {bm:.2f}  적용 {int(hi_te.sum()):5d}행   "
                  f"{gc:.4f} -> {gn:.4f}  Δ {gn-gc:+.4f}{flat}")
        sc = total_score(answer.loc[te], cur.loc[te], targets)[0]
        sn = total_score(answer.loc[te], new.loc[te], targets)[0]
        print(f"    총점            {sc:.4f} -> {sn:.4f}  Δ {sn-sc:+.4f}")

    print(BAR); print("판정 (사전 등록: 부호 3/3 ∧ 양수 ∧ |평균|>표준편차 ∧ 그룹 2개↑ 양수 ∧ PART 1 기전)")
    res = is_difference_real(df, answer, new, cur, targets, name_a="고풍속보정", name_b="현행")
    gpos = 0
    print("\n  [그룹별]")
    for g in targets:
        cap = CAPACITY_KWH[g]
        a = answer[g].to_numpy(float)
        gc = group_score(a, cur[g].to_numpy(float), cap)[0]
        gn = group_score(a, new[g].to_numpy(float), cap)[0]
        gpos += int(np.isfinite(gn - gc) and gn > gc)
        print(f"    {g:14s} {gc:.4f} -> {gn:.4f}  Δ {gn-gc:+.4f}")
    ok_all = bool(res and res["real"] and res["mean"] > 0 and gpos >= 2 and gate)
    print(f"\n  기전 {gate}  부호양수 {bool(res and res['mean']>0)}  양수그룹 {gpos}/{len(targets)}"
          f"  =>  {'✅ 제출 1회' if ok_all else '❌ 제출하지 않는다'}")

    if args.make_submission:
        print(BAR)
        tb = db / "raw_test_preds.csv"
        if not tb.exists():
            print(f"  ⚠ {tb} 없음.")
        elif not ok_all:
            print("  ⛔ 판정 미달. 제출 파일을 만들지 않는다.")
        else:
            print("  ⚠ test 의 풍속 축을 동일하게 만들어야 한다 (build_full_feature_pipeline 후 같은 컬럼).")
            print("    inference 경로 연결은 통과 확인 후 붙인다.")

    print(BAR); print(f"⏱ {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()