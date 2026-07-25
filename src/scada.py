"""
src/scada.py — SCADA 실측 활용 (가동률 보정 / 경험 파워커브 / 라벨 재구성)
==============================================================================
실측 head 로 확인한 사실
  * power_kw10m 은 kW 가 아니라 '10분간 kWh'. 시간값 = 6개 합.
    Vestas 정격 600 (=3.6MW), Unison 정격 ~708 (=4.2MW) 로 확인.
  * 시각 정렬: 라벨 H = SCADA kst_dtm 이 [H-0:50, H] 인 6행의 합  (offset +50분).
    둘 다 '구간 종료 시각' 이라 명세와 일치. R^2 = 0.99908
  * 계통 연계점 손실: label = 0.9869 * scada(G1), 0.9895 * scada(G2)
    잔차 RMSE 66~74 kWh = 설비용량의 0.31~0.34%. FICR 밴드(6%)의 1/20.
    => SCADA 로 라벨을 사실상 완전 재구성 가능 -> 가동률 판정이 확정적이 된다.

⚠️ SCADA 는 학습기간에만 존재한다. 테스트 피처로 쓰면 규칙 위반.
   여기서 만드는 것들은 전부 '학습 타깃을 정제하거나 정적 곡선을 적합' 하는 용도다.
==============================================================================
"""
import numpy as np
import pandas as pd

ALIGN_MIN = {"vestas": 50, "unison": 60}    # 기종별 검증된 정렬 오프셋(분)
# Vestas 는 10분 구간의 '끝', Unison 은 '시작' 을 찍는다. 풀 데이터 검증 결과
#   vestas +50분 R^2=0.99930  /  unison +60분 R^2=0.99848
# 같은 값을 쓰면 unison 이 10분 어긋나 group_3 손실계수 RMSE 가 5배로 뛴다.
RATED_KW10M = {"vestas": 600.0, "unison": 700.0}
TURB_CAP_KW = {"vestas": 3600.0, "unison": 4200.0}

# KPX 그룹 <-> 터빈 매핑 (info.xlsx)
GROUP_TURBINES = {
    "kpx_group_1": [("vestas", i) for i in range(1, 7)],
    "kpx_group_2": [("vestas", i) for i in range(7, 13)],
    "kpx_group_3": [("unison", i) for i in range(1, 6)],
}
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


def _cols(maker, i):
    return (f"{maker}_wtg{i:02d}_power_kw10m", f"{maker}_wtg{i:02d}_ws", f"{maker}_wtg{i:02d}_wd")


# ------------------------------------------------------------------ 로드/정제
def load_scada(path_or_df, maker, clip_power=True):
    """비물리적 센서값 정제. EDA 에서 본 ±4천만 급 이상치를 물리 범위로 자른다."""
    df = pd.read_csv(path_or_df, encoding="utf-8-sig") if isinstance(path_or_df, str) else path_or_df.copy()
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    rated = RATED_KW10M[maker]
    n_turb = 12 if maker == "vestas" else 5
    for i in range(1, n_turb + 1):
        pc, wc, dc = _cols(maker, i)
        if pc in df:
            bad = (df[pc] < -rated * 0.1) | (df[pc] > rated * 1.15)
            df.loc[bad, pc] = np.nan
            if clip_power:
                df[pc] = df[pc].clip(0, rated * 1.05)
        if wc in df:
            df.loc[(df[wc] < 0) | (df[wc] > 40), wc] = np.nan
    return df


def verify_alignment(scada, labels, maker, group, search=range(-6, 7), min_match=50):
    """
    풀 데이터에서 정렬을 다시 검증한다. head 로 정한 +50분이 전체에서도 맞는지 확인용.
    반환: DataFrame[offset_min, r2, rmse]  (r2 최대인 행이 정답)
    """
    lab = labels.copy(); lab["kst_dtm"] = pd.to_datetime(lab["kst_dtm"])
    pcs = [_cols(m, i)[0] for m, i in GROUP_TURBINES[group]]
    pcs = [c for c in pcs if c in scada.columns]
    rows = []
    for k in search:
        s = scada.copy()
        s["_h"] = (s["kst_dtm"] + pd.Timedelta(minutes=10 * k)).dt.floor("h")
        cnt = s.groupby("_h").size()
        agg = s.groupby("_h")[pcs].sum().loc[cnt[cnt == 6].index]
        m = agg.join(lab.set_index("kst_dtm")[[group]], how="inner").dropna()
        if len(m) < min_match:
            continue
        a, b = m[pcs].sum(axis=1).to_numpy(), m[group].to_numpy()
        rows.append({"offset_min": 10 * k, "r2": 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum(),
                     "rmse": float(np.sqrt(((a - b) ** 2).mean())), "n": len(m)})
    if not rows:
        raise ValueError(f"매칭된 시간이 min_match={min_match} 미만이다. "
                         f"데이터 범위가 겹치는지, min_match 를 낮춰야 하는지 확인할 것.")
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def to_hourly(scada, maker, align_min=None, require_full=True):
    """10분 kWh -> 시간 kWh (터빈별). ws/wd 는 평균/원형평균."""
    if align_min is None:
        align_min = ALIGN_MIN[maker]
    df = scada.copy()
    df["_h"] = (df["kst_dtm"] + pd.Timedelta(minutes=align_min)).dt.floor("h")
    n_turb = 12 if maker == "vestas" else 5
    p_cols = [_cols(maker, i)[0] for i in range(1, n_turb + 1) if _cols(maker, i)[0] in df]
    w_cols = [_cols(maker, i)[1] for i in range(1, n_turb + 1) if _cols(maker, i)[1] in df]
    d_cols = [_cols(maker, i)[2] for i in range(1, n_turb + 1) if _cols(maker, i)[2] in df]

    g = df.groupby("_h")
    out = g[p_cols].sum()                                   # kWh 는 합
    out = out.join(g[w_cols].mean())                        # 풍속은 평균
    for c in d_cols:                                        # 풍향은 원형평균
        rad = np.radians(df[c].to_numpy())
        tmp = pd.DataFrame({"_h": df["_h"], "s": np.sin(rad), "c": np.cos(rad)}).groupby("_h").mean()
        out[c] = (np.degrees(np.arctan2(tmp["s"], tmp["c"])) % 360.0)
    out["_n_intervals"] = g.size()
    if require_full:
        out = out[out["_n_intervals"] == 6]
    return out.drop(columns="_n_intervals").reset_index().rename(columns={"_h": "kst_dtm"})


# ------------------------------------------------------------ 풍향계 보정
def detect_wd_offset(hourly, maker, min_ws=5.0):
    """
    터빈별 풍향계 설치/배선 오프셋을 단지 원형중앙값 대비로 추정.
    (head 에서 Vestas WTG12 가 나머지보다 ~180도 틀어져 있었다. Unison 은 음수 규약.)
    """
    n_turb = 12 if maker == "vestas" else 5
    dc = [_cols(maker, i)[2] for i in range(1, n_turb + 1) if _cols(maker, i)[2] in hourly]
    wc = [_cols(maker, i)[1] for i in range(1, n_turb + 1) if _cols(maker, i)[1] in hourly]
    D = np.radians(hourly[dc].to_numpy() % 360.0)
    ok = hourly[wc].mean(axis=1).to_numpy() >= min_ws            # 약풍에서는 풍향계가 부정확
    ref = np.arctan2(np.nanmean(np.sin(D[ok]), 1), np.nanmean(np.cos(D[ok]), 1))
    res = {}
    for j, c in enumerate(dc):
        d = D[ok, j] - ref
        off = np.degrees(np.arctan2(np.nanmean(np.sin(d)), np.nanmean(np.cos(d))))
        spread = np.degrees(np.sqrt(-2 * np.log(np.clip(
            np.hypot(np.nanmean(np.sin(d)), np.nanmean(np.cos(d))), 1e-9, 1))))
        res[c] = {"offset_deg": float(off), "circ_spread_deg": float(spread)}
    return pd.DataFrame(res).T.sort_values("offset_deg", key=lambda s: -s.abs())


def apply_wd_offset(hourly, offsets):
    out = hourly.copy()
    for c, r in offsets.iterrows():
        if c in out:
            out[c] = (out[c] - r["offset_deg"]) % 360.0
    return out


# ------------------------------------------------------------ 가동률
def fit_reference_curve(hourly, maker, n_bins=None, q=0.90, min_per_bin=None, verbose=True):
    """
    같은 기종 터빈을 전부 풀링해서 하나의 기준 파워커브를 적합한다.
    터빈 1대씩 적합하면 bin 당 표본이 모자라 곡선이 무너진다 (실측 head 로 확인).
    정지/출력제한 샘플에 곡선이 끌려가지 않도록 상단 분위수(q=0.9)를 쓴다.
    반환: (bin 중심 ws, 기대 kWh/h)
    """
    n_turb = 12 if maker == "vestas" else 5
    rated_h = TURB_CAP_KW[maker]
    ws, kwh = [], []
    for i in range(1, n_turb + 1):
        pc, wc, _ = _cols(maker, i)
        if pc in hourly and wc in hourly:
            ws.append(hourly[wc].to_numpy(float)); kwh.append(hourly[pc].to_numpy(float))
    if not ws:
        raise ValueError(f"{maker} 파워/풍속 컬럼을 찾을 수 없다")
    ws = np.concatenate(ws); kwh = np.concatenate(kwh)
    ok = np.isfinite(ws) & np.isfinite(kwh)
    ws, kwh = ws[ok], kwh[ok]

    if n_bins is None:
        n_bins = int(np.clip(len(ws) // 400, 12, 60))       # 표본 수에 맞춰 해상도 조절
    if min_per_bin is None:
        min_per_bin = max(5, int(len(ws) / n_bins * 0.05))

    edges = np.linspace(0, 30, n_bins + 1)
    idx = np.clip(np.digitize(ws, edges) - 1, 0, n_bins - 1)
    ref = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= min_per_bin:
            ref[b] = np.quantile(kwh[m], q)
    centers = (edges[:-1] + edges[1:]) / 2

    n_valid = int(np.isfinite(ref).sum())
    if n_valid < 5:                                          # 데이터 부족 -> 물리 곡선 대체
        if verbose:
            print(f"   ⚠️ {maker}: 유효 bin {n_valid}개뿐 -> 이론 파워커브로 대체 "
                  f"(표본 {len(ws)}개. 풀 데이터로 다시 돌릴 것)")
        V_CI, V_R, V_CO = 3.0, 12.0, 25.0
        ref = np.where(centers < V_CI, 0.0,
              np.where(centers < V_R, (centers**3 - V_CI**3) / (V_R**3 - V_CI**3),
              np.where(centers < V_CO, 1.0, 0.0))) * rated_h
        return centers, ref

    ref = pd.Series(ref).interpolate(limit_direction="both").to_numpy()
    ref = np.clip(ref, 0, rated_h)
    hi = centers <= 24
    ref[hi] = np.maximum.accumulate(ref[hi])                 # cut-out 전까지만 단조
    ref[centers > 26] = 0.0
    if verbose:
        print(f"   {maker} 기준곡선: bin {n_valid}/{n_bins} 유효, 표본 {len(ws):,}개, "
              f"최대 {ref.max():.0f} kWh/h (정격 {rated_h:.0f})")
    return centers, ref


def availability_mask(hourly, maker, ws_thresh=4.5, dead_frac=0.03, derate_frac=0.55,
                      curve=None, verbose=True):
    """
    터빈-시간별 운전 상태 판정.
      stopped : 풍속 충분한데 출력이 기준곡선의 3% 미만  -> 정지 (정비/고장)
      derated : 출력이 기준곡선의 55% 미만               -> 출력제한/부분고장
    기준곡선은 같은 기종 전체를 풀링해 만든 하나를 공유한다.
    반환: (상태 DataFrame, 가용 플래그 DataFrame)
    """
    c, ref = curve if curve is not None else fit_reference_curve(hourly, maker, verbose=verbose)
    n_turb = 12 if maker == "vestas" else 5
    rated_h = TURB_CAP_KW[maker]
    status, avail = {}, {}
    for i in range(1, n_turb + 1):
        pc, wc, _ = _cols(maker, i)
        if pc not in hourly:
            continue
        ws = hourly[wc].to_numpy(float)
        kwh = hourly[pc].to_numpy(float)
        exp = np.interp(ws, c, ref)
        stopped = (ws >= ws_thresh) & (kwh < np.maximum(exp * dead_frac, rated_h * 0.01))
        derated = (~stopped) & (ws >= ws_thresh) & (kwh < exp * derate_frac)
        s = np.where(stopped, "stopped", np.where(derated, "derated", "ok")).astype(object)
        s[~np.isfinite(ws) | ~np.isfinite(kwh)] = "missing"
        status[f"{maker}_wtg{i:02d}"] = s
        avail[f"{maker}_wtg{i:02d}"] = (s == "ok") | (s == "derated")
    idx = hourly["kst_dtm"] if "kst_dtm" in hourly else hourly.index
    return (pd.DataFrame(status, index=idx), pd.DataFrame(avail, index=idx))


def group_available_capacity(avail_dict):
    """
    avail_dict : {"vestas": 가용플래그DF, "unison": 가용플래그DF}
    반환: DataFrame[kst_dtm, kpx_group_1..3]  = 그 시간의 가용 설비용량(kWh/h)
    """
    frames = []
    for group, turbs in GROUP_TURBINES.items():
        cols, caps = [], []
        for maker, i in turbs:
            key = f"{maker}_wtg{i:02d}"
            if maker in avail_dict and key in avail_dict[maker].columns:
                cols.append(avail_dict[maker][key]); caps.append(TURB_CAP_KW[maker])
        if not cols:
            continue
        A = pd.concat(cols, axis=1).astype(float)
        frames.append((A * np.array(caps)).sum(axis=1).rename(group))
    return pd.concat(frames, axis=1).reset_index().rename(columns={"index": "kst_dtm"})


# ------------------------------------------------------------ 타깃 정제
def fit_loss_factor(labels, scada_hourly_sum, groups=None):
    """label = k * (SCADA 합) 의 k 를 그룹별로 적합 (계통 연계점 손실)."""
    groups = groups or list(GROUP_TURBINES)
    out = {}
    for g in groups:
        if g not in labels or g not in scada_hourly_sum:
            continue
        m = labels[[g]].join(scada_hourly_sum[[g]], lsuffix="_lab", rsuffix="_sc").dropna()
        x, y = m[f"{g}_sc"].to_numpy(), m[f"{g}_lab"].to_numpy()
        ok = x > CAPACITY_KWH[g] * 0.05
        k = float(np.median(y[ok] / x[ok]))
        out[g] = {"k": k, "rmse": float(np.sqrt(((y - k * x) ** 2).mean())),
                  "rmse_pct_cap": float(np.sqrt(((y - k * x) ** 2).mean()) / CAPACITY_KWH[g] * 100)}
    return out


def clean_target(labels, avail_cap, loss=None, min_avail_frac=0.5):
    """
    가동률로 정규화한 학습 타깃.
        cf_clean = label / (k * 가용용량)
    정비/고장으로 눌린 시간이 '바람이 약했던 시간' 으로 오학습되는 걸 막는다.
    추론 시에는  pred_label = pred_cf * k * 설비용량 * 기대가용률  로 되돌린다.
    """
    lab = labels.copy()
    lab["kst_dtm"] = pd.to_datetime(lab["kst_dtm"])
    ac = avail_cap.copy(); ac["kst_dtm"] = pd.to_datetime(ac["kst_dtm"])
    m = lab.merge(ac, on="kst_dtm", how="left", suffixes=("", "_avail"))
    out = m[["kst_dtm"]].copy()
    for g in GROUP_TURBINES:
        if g not in m or f"{g}_avail" not in m:
            continue
        k = (loss or {}).get(g, {}).get("k", 1.0)
        cap_av = m[f"{g}_avail"].to_numpy(float)
        frac = cap_av / CAPACITY_KWH[g]
        cf = m[g].to_numpy(float) / np.where(cap_av > 0, k * cap_av, np.nan)
        cf[frac < min_avail_frac] = np.nan          # 절반 이상 멈춘 시간은 학습에서 제외
        out[f"{g}_cf"] = np.clip(cf, 0, 1.15)
        out[f"{g}_avail_frac"] = frac
    return out


# ------------------------------------------------------------ 정지 원인 분류
def refine_outage_cause(status, hourly, maker, co_frac=0.5, ws_high=17.0):
    """
    'stopped' 를 두 종류로 가른다. 이 구분을 안 하면 타깃 정제가 오히려 해가 된다.

      stopped_weather : 같은 기종에서 여러 대가 '동시에' 멈췄고 풍속이 높다
                        -> 고풍속 컷아웃/스톰 제어. 이건 기상에서 예측 가능하므로
                           타깃에서 제거하면 안 된다 (모델이 배워야 할 현상).
      stopped_maint   : 산발적 개별 정지 -> 정비/고장. 예측 불가이므로 정규화 대상.
    """
    st = status.copy()
    n_turb = st.shape[1]
    stopped = (st == "stopped")
    frac = stopped.sum(axis=1) / max(n_turb, 1)
    wcols = [_cols(maker, i)[1] for i in range(1, n_turb + 1) if _cols(maker, i)[1] in hourly]
    ws = (hourly.set_index("kst_dtm")[wcols].max(axis=1) if "kst_dtm" in hourly
          else hourly[wcols].max(axis=1)).reindex(st.index)
    weather = (frac >= co_frac) & (ws >= ws_high)
    for c in st.columns:
        m = stopped[c] & weather
        st.loc[m, c] = "stopped_weather"
        st.loc[stopped[c] & ~weather, c] = "stopped_maint"
    return st


def availability_from_status(status):
    """정비 정지만 '가용 아님' 으로 본다. 고풍속 정지는 예측 대상이므로 가용으로 유지."""
    return ~status.isin(["stopped_maint", "missing"])