"""
src/postprocessing.py  (개선판 — 기존 파일 drop-in 교체용)
==============================================================================
기존 대비 바뀐 점
  1) [버그수정] Nelder-Mead 초기 simplex 스케일 붕괴 해결
     scipy는 x0 성분이 0이면 0.00025, 아니면 x0*1.05 로 simplex를 만든다.
     기존 initial_guess=[1.0, 0.0, 10.0] 은 설비용량이 21,600 kWh 인데
     beta 를 0.00025 kWh, zero_th 를 0.5 kWh 스텝으로 탐색하게 만든다.
     => 사실상 alpha 한 개만 최적화되고 있었음. 합성 OOF 실험에서 +0.010 손실.
  2) 다중 시작점(multi-start) 으로 계단형 목적함수의 국소최적 탈출
  3) [신규] 구간별(piecewise) 단조 보정 — alpha/beta 전역 선형보다 표현력이 큼
  4) [신규] fit/eval 분리 — 같은 OOF 에 fit 하고 같은 OOF 로 평가하면
     보고되는 개선폭이 in-sample 과대추정이 된다.
==============================================================================
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from src.utils import CAPACITY_KWH, TARGET_COLS
except ImportError:                                       # 단독 실행/테스트용
    TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
    CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

KNOTS = np.array([0.0, 0.15, 0.35, 0.60, 0.85, 1.00])     # piecewise 보정 절점 (용량 정규화)


# ------------------------------------------------------------------ 점수 계산
def _price(e):
    return np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], default=0.0)


def _group_score(actual, forecast, cap):
    """
    단일 그룹의 0.5*(1-NMAE) + 0.5*FICR (그룹평균 1/3 은 상수라 생략).
    예측에 NaN 이 섞이면(정제 타깃 학습 시 제외된 행 등) 점수 전체가 NaN 이 되어
    최적화가 아무 후보도 못 고른다. 반드시 유한값만 남기고 계산한다.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = np.isfinite(a) & np.isfinite(f)
    if ok.sum() == 0:
        return 0.0
    a, f = a[ok], f[ok]
    v = a >= cap * 0.10
    if v.sum() == 0:
        return 0.0
    a, f = a[v], f[v]
    er = np.abs(f - a) / cap
    return 0.5 * (1.0 - er.mean()) + 0.5 * (a * _price(er)).sum() / (a * 4.0).sum()


# ------------------------------------------------------------------ 보정 적용
def _transform_linear(y, alpha, beta, zero_th, cap):
    adj = y * alpha + beta
    adj = np.where(adj < zero_th, 0.0, adj)
    return np.clip(adj, 0.0, cap)


def _transform_piecewise(y, out_knots, zero_th, cap):
    """
    용량 정규화된 예측값 p=y/cap 을 절점 KNOTS -> out_knots 로 단조 매핑.
    저출력/중출력/고출력 구간마다 다른 보정을 걸 수 있어 밴드 진입에 유리하다.
    """
    ok = np.maximum.accumulate(np.asarray(out_knots, dtype=float))   # 단조 강제
    p = np.clip(y / cap, 0.0, 1.0)
    adj = np.interp(p, KNOTS, ok) * cap
    adj = np.where(adj < zero_th, 0.0, adj)
    return np.clip(adj, 0.0, cap)


def apply_postprocessing(pred_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """기존 시그니처 유지. params 에 'out_knots' 가 있으면 piecewise, 없으면 기존 선형."""
    out = pred_df.copy()
    for col in TARGET_COLS:
        if col not in params or col not in out.columns:
            continue
        p = params[col]
        cap = CAPACITY_KWH[col]
        y = out[col].to_numpy(dtype=float)
        if "out_knots" in p:
            out[col] = _transform_piecewise(y, p["out_knots"], p.get("zero_th", 0.0), cap)
        else:
            out[col] = _transform_linear(y, p.get("alpha", 1.0), p.get("beta", 0.0),
                                         p.get("zero_th", 0.0), cap)
    return out


# ------------------------------------------------------------------ 최적화
def _optimize_one(y_true, y_pred, cap, mode="piecewise", n_restarts=None, seed=42):
    """단일 그룹 파라미터 탐색. 스케일 맞춘 initial_simplex + multi-start."""
    rng = np.random.default_rng(seed)

    # 평가 대상이 될 수 있는 행이 아예 없으면 최적화가 무의미하다.
    # (모든 후보가 동점이 되어 '첫 후보' 가 뽑히는 사고를 막는다)
    _a = np.asarray(y_true, float); _f = np.asarray(y_pred, float)
    _ok = np.isfinite(_a) & np.isfinite(_f)
    if _ok.sum() == 0 or (_a[_ok] >= cap * 0.10).sum() < 30:
        print(f"   ⚠️ 평가 가능한 행이 {int((_a[_ok] >= cap*0.10).sum()) if _ok.sum() else 0}개뿐 "
              f"-> 후처리 생략(항등변환)")
        ident = (np.array([1.0, 0.0, 0.0]) if mode == "linear" else np.r_[KNOTS, 0.0])
        return ident, _group_score(y_true, y_pred, cap)

    if mode == "linear":
        def neg(x):
            return -_group_score(y_true, _transform_linear(y_pred, x[0], x[1], x[2], cap), cap)

        starts = [(a, b * cap, z * cap)
                  for a in (0.90, 1.00, 1.10)
                  for b in (-0.04, 0.0, 0.04, 0.08)
                  for z in (0.0, 0.08, 0.15)]
        steps = np.array([0.10, 0.05 * cap, 0.06 * cap])       # <- 핵심: 스케일 일치
    else:
        def neg(x):
            return -_group_score(y_true, _transform_piecewise(y_pred, x[:-1], x[-1], cap), cap)

        starts = [tuple(KNOTS + s) + (z * cap,)
                  for s in (-0.04, 0.0, 0.04)
                  for z in (0.0, 0.08, 0.15)]
        steps = np.r_[np.full(len(KNOTS), 0.05), 0.06 * cap]

    if n_restarts is not None:
        starts = list(starts)[:n_restarts]

    best_x, best_f = None, np.inf
    for s in starts:
        x0 = np.asarray(s, dtype=float)
        sim = np.vstack([x0] + [x0 + np.eye(len(x0))[k] * steps[k] for k in range(len(x0))])
        r = minimize(neg, x0, method="Nelder-Mead",
                     options={"initial_simplex": sim, "maxiter": 6000,
                              "xatol": 1e-4, "fatol": 1e-9})
        if np.isfinite(r.fun) and r.fun < best_f:
            best_f, best_x = r.fun, r.x
    if best_x is None:                       # 전 후보가 실패 -> 항등변환으로 안전 복귀
        best_x = (np.array([1.0, 0.0, 0.0]) if mode == "linear"
                  else np.r_[KNOTS, 0.0])
        best_f = -_group_score(y_true, y_pred, cap)
        print("   ⚠️ 후처리 최적화가 유효한 해를 못 찾았다 -> 항등변환 유지 "
              "(예측에 NaN 이 많거나 평가대상 행이 부족한 경우)")
    return best_x, -best_f


def optimize_postprocessing(answer_df: pd.DataFrame, oof_pred_df: pd.DataFrame,
                            mode: str = "piecewise", eval_mask=None, verbose: bool = True) -> dict:
    """
    기존 시그니처 호환. mode='linear' 로 두면 기존 alpha/beta/zero_th 형태(단, 스케일 수정본).

    eval_mask : bool Series/array. 주면 ~eval_mask 구간에서만 파라미터를 fit 하고
                eval_mask 구간 점수를 정직한 out-of-sample 개선폭으로 함께 출력한다.
                (예: eval_mask = 연도가 2024인 행)  <- 이거 꼭 쓰길 권함
    """
    best_params = {}
    for col in TARGET_COLS:
        m = answer_df[col].notna().to_numpy()
        y_true = answer_df[col].to_numpy(dtype=float)
        y_pred = oof_pred_df[col].to_numpy(dtype=float)
        cap = CAPACITY_KWH[col]

        if eval_mask is None:
            fit_m, ev_m = m, None
        else:
            em = np.asarray(eval_mask, dtype=bool)
            fit_m, ev_m = m & ~em, m & em

        x, fit_score = _optimize_one(y_true[fit_m], y_pred[fit_m], cap, mode=mode)

        if mode == "linear":
            best_params[col] = {"alpha": float(x[0]), "beta": float(x[1]),
                                "zero_th": float(max(0.0, x[2]))}
        else:
            best_params[col] = {"out_knots": np.maximum.accumulate(x[:-1]).tolist(),
                                "zero_th": float(max(0.0, x[-1]))}

        if verbose:
            msg = f"  ├─ {col}: fit {fit_score:.4f}"
            if ev_m is not None and ev_m.sum() > 0:
                pp = (_transform_linear(y_pred[ev_m], *x, cap) if mode == "linear"
                      else _transform_piecewise(y_pred[ev_m], x[:-1], x[-1], cap))
                before = _group_score(y_true[ev_m], y_pred[ev_m], cap)
                after = _group_score(y_true[ev_m], pp, cap)
                msg += f" | holdout {before:.4f} -> {after:.4f} ({after - before:+.4f})"
            print(msg)
    return best_params