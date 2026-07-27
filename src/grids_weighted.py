"""
src/grids_weighted.py — A-4 격자 가중 (v21)
==============================================================================
[진행] 통합(long) 경로에 **폴드별·그룹별 가중 허브고도 풍속** 3컬럼을 명시적으로 준다.

왜 이 형태인가 (§8.3 A-4 의 원안을 수정한 근거)
  원안은 "`ldaps_mean_*`(16격자 균등평균)을 데이터 기반 가중평균으로 **대체**" 였음.
  착수 전 §3.17 규칙대로 기존 선택 결과를 조회했더니 전제가 사실과 달랐음:

      v13 폴드별 top-200 평균 (11폴드)
        ldaps_mean_*        3.9 개  (24개 중)   <- A-4 가 고치려던 대상
        gfs_mean_*          9.3 개
        LDAPS 개별 격자    83.1 개   <- 모델이 실제로 쓰는 것
        GFS  개별 격자     85.8 개

  즉 선택기는 §3.4 의 "개별 격자 > `_mean_`" 을 이미 실행하고 있었고,
  균등평균은 거의 버려져 있었음. 대체형으로는 폴드당 3.9컬럼만 바뀌어 무의미함.

  그러나 §4.4 가 잰 사실(동일 시각 10m U성분 G1 5.378 / G3 6.072 / 균등평균 5.298)은
  여전히 유효함. 모델이 그룹별 가중 풍속을 스스로 만들려면
  `w1·g1 + ... + w16·g16` 을 **그룹 조건부**로 학습해야 하는데, 이는
  **선형결합 × 상호작용**이고 §3.2 가 "트리가 공짜로 못 얻는 것" 으로 지목한 형태임.
  => 명시적으로 3컬럼만 준다.

설계
  * 통합 경로 전용. 행이 (시각 × 그룹)이라 **그룹마다 다른 값**을 넣을 수 있음.
  * G2 는 독립 경로가 예측하므로 **손대지 않음 = 무손상 대조군**
    (--featcols-from 으로 선택까지 고정하면 G2 OOF 는 v13 과 비트 단위로 같아야 함).
  * 가중치는 **각 폴드의 학습 행에서만** 산출 (선택 누설 차단, §3.3 과 같은 원칙).
  * 비음수 최소제곱(NNLS)으로 16격자 풍력밀도 대리량 -> 그룹 타깃. 16파라미터 / 학습행 8.6k~17k.
  * 산출 컬럼 3개 (접두사 `wgrid_`):
        wgrid_ws   = Σ wᵢ·wsᵢ      가중 풍속
        wgrid_u    = Σ wᵢ·uᵢ       가중 U성분 (§3.4: 생존 피처가 전부 U성분이었음)
        wgrid_v    = Σ wᵢ·vᵢ
  * 컬럼 3개뿐이라 §3.21 의 선택예산 대체관계 위험이 사실상 없음.
    그래도 --force-prefix wgrid_ 로 강제 포함해 '선택에서 떨어져 미측정' (v20 사고)을 막음.

⚠ v20b 와의 차이 (같은 함정인지 점검)
  v20b 는 **블록 상수**(24행 동일값)라 날짜 식별자로 작동한 것이 폐기 원인 가설이었음.
  wgrid_* 는 시각마다 값이 변하는 물리량이고 기존 컬럼들의 선형결합이라 기전이 다름.
  다만 '강제 포함' 이라는 절차는 같으므로, 게이트 미달 시 축을 바로 닫는다(사전등록).
==============================================================================
"""
import re

import numpy as np
import pandas as pd

N_GRID = 16
U10 = "ldaps_g{i}_heightAboveGround_10_10u"
V10 = "ldaps_g{i}_heightAboveGround_10_10v"
OUT_COLS = ["wgrid_ws", "wgrid_u", "wgrid_v"]


def grid_uv(df):
    """16격자의 10m U/V 를 (n, 16) 배열 두 개로 꺼낸다. 없으면 None."""
    us, vs = [], []
    for i in range(1, N_GRID + 1):
        cu, cv = U10.format(i=i), V10.format(i=i)
        if cu not in df.columns or cv not in df.columns:
            return None, None
        us.append(df[cu].to_numpy(np.float64))
        vs.append(df[cv].to_numpy(np.float64))
    return np.column_stack(us), np.column_stack(vs)


def fit_weights(ws, y, fit_mask):
    """비음수 최소제곱으로 16격자 가중치를 구한다. 학습 행만 씀.

    풍력밀도가 ws³ 에 비례하므로 설계행렬은 ws³ 로 둔다(단조변환이 아니라
    **결합 계수**를 정하는 문제라 §3.2 의 불변성과 무관함).
    반환: (16,) 합이 1인 비음수 가중치. 실패하면 균등 가중.
    """
    from scipy.optimize import nnls

    m = fit_mask & np.isfinite(y) & np.isfinite(ws).all(axis=1)
    if m.sum() < 500:
        return np.full(N_GRID, 1.0 / N_GRID)
    A = ws[m] ** 3
    s = A.mean()
    if not np.isfinite(s) or s <= 0:
        return np.full(N_GRID, 1.0 / N_GRID)
    try:
        w, _ = nnls(A / s, y[m])
    except Exception:
        return np.full(N_GRID, 1.0 / N_GRID)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.full(N_GRID, 1.0 / N_GRID)
    return w / tot


def make_fold_feature_fn(targets, verbose=True):
    """run_cv_joint 에 넘길 콜백을 만든다.

    콜백 규약: fn(long_df, train_row_mask, fold_name) -> DataFrame(index=long_df.index,
    columns=OUT_COLS). long_df 는 build_long 결과이고 `_gname`, `_y` 를 갖고 있다.
    """
    def fn(long_df, train_mask, fold_name):
        u, v = grid_uv(long_df)
        if u is None:
            raise KeyError("ldaps_g{1..16}_heightAboveGround_10_{10u,10v} 가 없음 "
                           "— A-4 는 LDAPS 개별 격자 컬럼을 요구함")
        ws = np.sqrt(u * u + v * v)
        y = long_df["_y"].to_numpy(np.float64)
        gname = long_df["_gname"].to_numpy()

        out = pd.DataFrame(index=long_df.index, columns=OUT_COLS, dtype=np.float32)
        for g in targets:
            gm = (gname == g)
            w = fit_weights(ws, y, train_mask & gm)
            out.loc[gm, "wgrid_ws"] = (ws[gm] @ w).astype(np.float32)
            out.loc[gm, "wgrid_u"] = (u[gm] @ w).astype(np.float32)
            out.loc[gm, "wgrid_v"] = (v[gm] @ w).astype(np.float32)
            if verbose:
                top = np.argsort(-w)[:4]
                txt = " ".join(f"g{i+1}:{w[i]:.2f}" for i in top)
                print(f"   ·  [{fold_name}] {g:<13s} 상위격자 {txt}  "
                      f"(유효 {int((w > 0.01).sum())}/16, 균등대비 최대 {w.max()*N_GRID:.1f}배)")
        return out.astype(np.float32)

    return fn
