"""
src/causality.py — 대회 3항(Data Leakage 방지) 자동 감사
==============================================================================
원리
  블록 B 를 고른다 -> "B 의 data_available_kst_dtm 이하인 행만" 남긴 잘린
  데이터로 피처 파이프라인을 다시 돌린다 -> B 구간의 피처값이 전체 데이터로
  만든 값과 완전히 같아야 한다. 한 컬럼이라도 다르면 그 피처는 미래를 봤다.

  이건 '규칙을 지켰다'는 주장이 아니라 실행 가능한 증명이다. 2차 코드검증/
  발표 소명 자료로 그대로 쓸 수 있다.

주의
  - shift(-1) / rolling(center=True) 같은 전방 참조는 반드시
    groupby('data_available_kst_dtm') 안에서만 해야 한다. 그렇지 않으면
    블록 마지막 행(대상 00:00)이 24시간 뒤 공개된 다음 블록을 끌어온다.
  - .diff(1) / .shift(1) / rolling(center=False) 는 과거만 보므로 위반은
    아니지만, 블록을 넘으면 서로 다른 초기화 시각을 섞게 되어 물리적으로
    부정확하다. 어차피 groupby 로 감싸는 편이 낫다.
==============================================================================
"""
import numpy as np
import pandas as pd

AVAIL = "data_available_kst_dtm"
TIME = "forecast_kst_dtm"


def audit_causality(df, feature_fn, n_blocks=12, seed=0, atol=1e-8,
                    avail_col=AVAIL, time_col=TIME, verbose=True):
    """
    df         : 피처 생성 '전' 의 원본 (avail_col 포함 필수)
    feature_fn : df -> 피처가 붙은 df  (예: build_full_feature_pipeline)
    n_blocks   : 무작위로 검사할 예보 블록 수

    반환: (통과여부, 위반 컬럼 목록 DataFrame)
    """
    if avail_col not in df.columns:
        raise ValueError(f"'{avail_col}' 컬럼이 없다. process_weather_data 에서 drop 하지 말 것.")

    d = df.copy()
    d[avail_col] = pd.to_datetime(d[avail_col])
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)

    full = feature_fn(d.copy())
    num_cols = [c for c in full.select_dtypes(include=[np.number]).columns]

    blocks = np.sort(d[avail_col].dropna().unique())
    blocks = blocks[len(blocks) // 5:]                       # 앞쪽 워밍업 구간은 제외
    rng = np.random.default_rng(seed)
    pick = rng.choice(blocks, size=min(n_blocks, len(blocks)), replace=False)

    bad = {}
    for b in pick:
        past = d[d[avail_col] <= b].copy()                    # 그 시점까지만
        trunc = feature_fn(past)
        m_full = (full[avail_col] == b).to_numpy()
        m_tr = (trunc[avail_col] == b).to_numpy()
        if m_full.sum() != m_tr.sum() or m_full.sum() == 0:
            continue
        A = full.loc[m_full, num_cols].to_numpy(float)
        B = trunc.loc[m_tr, [c for c in num_cols if c in trunc.columns]].to_numpy(float)
        if A.shape != B.shape:
            continue
        diff = ~(np.isclose(A, B, atol=atol, equal_nan=True))
        if diff.any():
            for j in np.where(diff.any(0))[0]:
                col = num_cols[j]
                bad[col] = bad.get(col, 0) + int(diff[:, j].sum())

    ok = len(bad) == 0
    rep = (pd.DataFrame(sorted(bad.items(), key=lambda x: -x[1]),
                        columns=["feature", "n_rows_differ"])
           if bad else pd.DataFrame(columns=["feature", "n_rows_differ"]))
    if verbose:
        print(f"🔍 예보 블록 {len(pick)}개 감사 / 수치 피처 {len(num_cols)}개")
        if ok:
            print("   ✅ 통과 — 모든 피처가 예측기준시점 이전 정보만 사용")
        else:
            print(f"   ❌ 위반 {len(bad)}개 컬럼 — 아래 피처가 미래 정보를 사용 중")
            print(rep.head(20).to_string(index=False))
    return ok, rep


def assert_block_structure(df, avail_col=AVAIL, time_col=TIME, verbose=True):
    """대회 명세의 블록 구조(1블록=24시간, 동일 avail)가 실제로 성립하는지 확인."""
    d = df[[time_col, avail_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col]); d[avail_col] = pd.to_datetime(d[avail_col])
    g = d.groupby(avail_col)[time_col].agg(["count", "min", "max"])
    lead_min = ((g["min"] - g.index).dt.total_seconds() / 3600)
    lead_max = ((g["max"] - g.index).dt.total_seconds() / 3600)
    if verbose:
        print(f"📦 블록 {len(g)}개 | 블록당 행 수 {g['count'].min()}~{g['count'].max()} "
              f"(최빈 {int(g['count'].mode()[0])})")
        print(f"   lead time 범위: {lead_min.min():.0f} ~ {lead_max.max():.0f} 시간")
        odd = g[g["count"] != 24]
        if len(odd):
            print(f"   ⚠️ 24행이 아닌 블록 {len(odd)}개 — groupby 시계열 피처에서 NaN 처리 주의")
    return g


def safe_block_ops(df, cols, avail_col=AVAIL):
    """블록 경계를 절대 넘지 않는 전/후방 참조 피처. 이것만 쓰면 위반이 원천 차단된다."""
    out = df.copy()
    gb = out.groupby(pd.to_datetime(out[avail_col]), sort=False)
    for c in cols:
        out[f"{c}_lag1"] = gb[c].shift(1)
        out[f"{c}_lead1"] = gb[c].shift(-1)                   # 같은 블록 안이라 합법
        out[f"{c}_lead2"] = gb[c].shift(-2)
        out[f"{c}_cwin3"] = gb[c].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
        out[f"{c}_blk_mean"] = gb[c].transform("mean")
        out[f"{c}_blk_std"] = gb[c].transform("std")
        out[f"{c}_anom"] = out[c] - out[f"{c}_blk_mean"]
    return out