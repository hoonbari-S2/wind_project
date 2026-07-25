"""
src/validation.py
==============================================================================
BARAM 2026 - 검증 체계 모듈

해결하는 문제 2가지
  [1] Early stopping 누설
      기존 train.py 는 eval_set=[(X_val, y_val)] 로 조기종료 시점을 정하고
      바로 그 X_val 로 OOF 를 만들었다. OOF 가 낙관적으로 부풀고, 모델마다
      부푸는 정도가 달라서(LightGBM 은 callbacks=[] 라 ES 자체가 없었음)
      OOF 기준 앙상블 가중치가 "가장 많이 샌 모델"로 쏠린다.
      => fold 내부에서 예보블록(day) 단위로 ES 전용 셋을 따로 떼고,
         best_iteration 을 얻은 뒤 fold 전체 학습셋으로 재학습(refit)한다.

  [2] 검증 구성이 테스트 상황과 다름
      테스트는 2025년 '통째로'인데 기존 GroupKFold(월 단위)는 3년치에서
      흩어진 월을 검증한다. 같은 계절의 다른 해 데이터가 항상 학습에 있어
      "미지의 1년" 난이도를 재현하지 못한다.
      => LOYO(Leave-One-Year-Out). 각 fold 가 나머지 연도로 학습해
         한 해 전체를 예측한다. 테스트 구조와 동일.

용어
  fday : 예보 블록 키. 예보는 매일 09KST 초기화 -> 13KST 공개 -> 익일 01시~
         익익일 00시. 즉 24시간이 한 덩어리로 같이 공개된다. 데이터를 쪼갤 때
         (ES split 포함) 반드시 이 덩어리 단위로 잘라야 한다. 시간 단위로
         랜덤 분할하면 인접 시각 상관이 0.95라 ES 셋이 학습셋과 사실상 동일해져
         조기종료가 아예 작동하지 않는다.
==============================================================================
"""
import numpy as np
import pandas as pd

try:
    from src.utils import CAPACITY_KWH, TARGET_COLS
except ImportError:
    TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
    CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


# ----------------------------------------------------------------- 시간 키
def add_time_keys(df, time_col="forecast_kst_dtm", avail_col="data_available_kst_dtm"):
    """_year(연도) 와 _fday(예보 블록 키) 를 붙인다."""
    out = df.copy()
    t = pd.to_datetime(out[time_col])
    out["_year"] = t.dt.year.to_numpy()
    if avail_col in out.columns and out[avail_col].notna().any():
        out["_fday"] = pd.to_datetime(out[avail_col]).dt.normalize().to_numpy()
    else:
        # avail 컬럼이 없으면 블록 구조로 역산:
        # 01:00~23:00(D+1) 과 00:00(D+2) 이 한 블록 -> (t - 1h).date 로 묶인다
        out["_fday"] = (t - pd.Timedelta(hours=1)).dt.normalize().to_numpy()
    return out


# ----------------------------------------------------------------- 폴드 생성
def make_folds(years, fdays, scheme="loyo", n_splits=5, seed=42):
    """
    scheme
      'loyo'         : 연 단위 Leave-One-Year-Out  <- 권장. 테스트(2025 전체)와 동일 구조
      'holdout2024'  : 2024년 단일 홀드아웃 (빠른 확인용)
      'block_month'  : 연속된 월 블록 K-fold (셔플 없음)
      'month_group'  : 기존 코드 재현용 (sklearn GroupKFold, 월 그룹)
    반환: [(train_idx, val_idx, fold_name), ...]
    """
    years = np.asarray(years)
    idx = np.arange(len(years))
    folds = []

    if scheme == "loyo":
        for y in sorted(pd.unique(years)):
            va = idx[years == y]
            tr = idx[years != y]
            if len(va) == 0 or len(tr) == 0:
                continue
            folds.append((tr, va, f"val{y}"))

    elif scheme == "holdout2024":
        va, tr = idx[years == 2024], idx[years != 2024]
        if len(va) and len(tr):
            folds.append((tr, va, "val2024"))

    elif scheme == "block_month":
        per = pd.PeriodIndex(pd.to_datetime(fdays), freq="M")
        uniq = np.array(sorted(per.unique().astype(str)))
        chunks = np.array_split(uniq, n_splits)          # 연속 월 블록
        pstr = per.astype(str).to_numpy()
        for k, ch in enumerate(chunks):
            m = np.isin(pstr, ch)
            folds.append((idx[~m], idx[m], f"block{k}"))

    elif scheme == "month_group":
        from sklearn.model_selection import GroupKFold
        g = pd.PeriodIndex(pd.to_datetime(fdays), freq="M").astype(str)
        for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(idx, groups=g)):
            folds.append((tr, va, f"gkf{k}"))
    else:
        raise ValueError(f"알 수 없는 scheme: {scheme}")

    return folds


def inner_es_split(fdays_tr, frac=0.15, seed=0):
    """fold 학습셋 안에서 ES 전용 셋을 '예보블록 단위'로 떼어낸다."""
    days = pd.unique(fdays_tr)
    rng = np.random.default_rng(seed)
    n_es = max(1, int(round(len(days) * frac)))
    es_days = set(pd.Series(rng.choice(days, size=n_es, replace=False)).tolist())
    is_es = np.array([d in es_days for d in fdays_tr])
    return ~is_es, is_es


# ----------------------------------------------------------------- 모델 래퍼
def _build(model_type, params):
    if model_type == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    if model_type == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    if model_type == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    if model_type == "SklearnGBM":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params)
    if model_type in ("RandomForest", "ExtraTrees"):
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        return (RandomForestRegressor if model_type == "RandomForest" else ExtraTreesRegressor)(**params)
    raise ValueError(f"지원하지 않는 model_type: {model_type}")


def _best_iter(model, model_type, fallback):
    try:
        if model_type == "XGBoost":
            return int(model.best_iteration) + 1
        if model_type == "LightGBM":
            return int(model.best_iteration_)
        if model_type == "CatBoost":
            return int(model.get_best_iteration()) + 1
        if model_type == "SklearnGBM":
            return int(model._es_best_iter)
    except Exception:
        pass
    return fallback


def fit_no_leak(model_type, params, X_tr, y_tr, fdays_tr,
                es_rounds=50, es_frac=0.15, mode="refit", seed=0, verbose=False):
    """
    누설 없는 학습.

    mode
      'refit'  : 내부 ES 셋으로 best_iteration 을 찾고, 그 값으로 fold 학습셋
                 '전체'를 다시 학습. 데이터 손실 없음 + 검증셋 미접촉. (권장)
      'inner'  : 내부 ES 셋을 그대로 두고 한 번만 학습 (빠르지만 15% 손실)
      'fixed'  : ES 없이 config 의 n_estimators 그대로
    """
    p = dict(params)
    n_max = int(p.get("n_estimators", p.get("iterations", 1000)) or 1000)

    if mode == "fixed":
        m = _build(model_type, p)
        m.fit(X_tr, y_tr)
        return m, n_max

    tr_m, es_m = inner_es_split(fdays_tr, frac=es_frac, seed=seed)
    X_a, y_a = X_tr.iloc[tr_m], y_tr.iloc[tr_m]
    X_e, y_e = X_tr.iloc[es_m], y_tr.iloc[es_m]

    p_es = dict(p)
    if model_type == "XGBoost":
        p_es["early_stopping_rounds"] = es_rounds
        m = _build(model_type, p_es)
        m.fit(X_a, y_a, eval_set=[(X_e, y_e)], verbose=False)
    elif model_type == "LightGBM":
        import lightgbm as lgb
        m = _build(model_type, p_es)
        m.fit(X_a, y_a, eval_set=[(X_e, y_e)],
              callbacks=[lgb.early_stopping(es_rounds, verbose=False), lgb.log_evaluation(0)])
    elif model_type == "CatBoost":
        p_es["early_stopping_rounds"] = es_rounds
        m = _build(model_type, p_es)
        m.fit(X_a, y_a, eval_set=(X_e, y_e), verbose=False)
    elif model_type == "SklearnGBM":
        # sklearn GBM 은 eval_set 이 없으므로 staged_predict 로 직접 ES 곡선을 만든다
        m = _build(model_type, p_es)
        m.fit(X_a, y_a)
        errs = [np.abs(pr - y_e.to_numpy()).mean() for pr in m.staged_predict(X_e)]
        m._es_best_iter = int(np.argmin(errs)) + 1
    else:                                                   # RF/ET 는 ES 개념 없음
        m = _build(model_type, p)
        m.fit(X_tr, y_tr)
        return m, n_max

    bi = _best_iter(m, model_type, n_max)
    if mode == "inner":
        return m, bi

    # refit: 데이터가 1/(1-frac) 배로 늘었으니 트리 수도 그만큼 늘려서 전체 재학습
    n_final = int(np.clip(round(bi / (1.0 - es_frac)), 10, n_max))
    p_full = dict(p)
    p_full.pop("early_stopping_rounds", None)
    if model_type == "CatBoost":
        p_full["iterations"] = n_final
    else:
        p_full["n_estimators"] = n_final
    m_full = _build(model_type, p_full)
    m_full.fit(X_tr, y_tr)
    if verbose:
        print(f"      ES best_iter={bi} -> refit n={n_final}")
    return m_full, n_final


# ----------------------------------------------------------------- 점수
def _price(e):
    return np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], default=0.0)


def group_score(actual, forecast, cap):
    """단일 그룹 0.5*(1-NMAE)+0.5*FICR. 평가대상(>=10%cap)이 없으면 nan."""
    a = np.asarray(actual, float)
    f = np.asarray(forecast, float)
    ok = ~np.isnan(a) & ~np.isnan(f)
    a, f = a[ok], f[ok]
    v = a >= cap * 0.10
    if v.sum() == 0:
        return np.nan, np.nan, np.nan
    a, f = a[v], f[v]
    er = np.abs(f - a) / cap
    nmae = er.mean()
    ficr = (a * _price(er)).sum() / (a * 4.0).sum()
    return 0.5 * (1 - nmae) + 0.5 * ficr, 1 - nmae, ficr


def total_score(answer_df, pred_df, targets=None):
    """대회 공식 산식 (3그룹 평균)."""
    targets = targets or TARGET_COLS
    nm, fi = [], []
    for c in targets:
        s, one_m, f = group_score(answer_df[c].to_numpy(float),
                                  pred_df[c].to_numpy(float), CAPACITY_KWH[c])
        if np.isnan(s):
            continue
        nm.append(1 - one_m)
        fi.append(f)
    one_minus_nmae = 1 - np.mean(nm)
    ficr = np.mean(fi)
    return 0.5 * one_minus_nmae + 0.5 * ficr, one_minus_nmae, ficr


# ----------------------------------------------------------------- CV 실행
def run_cv(df, feature_cols, targets, model_type, model_params,
           scheme="loyo", n_splits=5, es_rounds=50, es_frac=0.15,
           es_mode="refit", seed=42, save_dir=None, verbose=True,
           predict_all=True):
    """
    반환
      oof     : DataFrame (df.index 정렬, 컬럼=targets). 누설 없는 OOF.
      fold_df : fold x target 별 점수표
      models  : {(target, fold_name): model}
    """
    import joblib
    df = add_time_keys(df)
    oof = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    rows, models = [], {}

    for target in targets:
        fit_mask = df[target].notna().to_numpy()          # 학습에 쓸 수 있는 행
        # predict_all=True 면 폴드는 '전체 행' 기준으로 나누고, 학습만 fit_mask 로 제한한다.
        # (정제 타깃처럼 일부 행이 학습에서 빠져도 예측은 모든 행에 나오도록)
        sub = df if predict_all else df.loc[fit_mask]
        base = np.arange(len(df))[fit_mask] if predict_all else None
        X = sub[feature_cols].reset_index(drop=True)
        y = sub[target].reset_index(drop=True)
        yrs, fds = sub["_year"].to_numpy(), sub["_fday"].to_numpy()
        sub_fit = fit_mask if predict_all else np.ones(len(sub), bool)
        cap = CAPACITY_KWH[target]
        folds = make_folds(yrs, fds, scheme=scheme, n_splits=n_splits, seed=seed)

        if verbose:
            print(f"\n🌀 [{target}] scheme={scheme}  folds={[f[2] for f in folds]}  "
                  f"학습가능 {int(sub_fit.sum()):,} / 예측대상 {len(sub):,}행")

        pred = np.full(len(sub), np.nan)
        for tr, va, name in folds:
            tr_fit = tr[sub_fit[tr]]                       # 타깃이 있는 학습 행만
            if len(tr_fit) < 100:
                continue
            model, n_used = fit_no_leak(model_type, model_params,
                                        X.iloc[tr_fit], y.iloc[tr_fit], fds[tr_fit],
                                        es_rounds=es_rounds, es_frac=es_frac,
                                        mode=es_mode, seed=seed, verbose=False)
            pred[va] = np.clip(model.predict(X.iloc[va]), 0, cap)
            models[(target, name)] = model
            if save_dir is not None:
                joblib.dump(model, f"{save_dir}/model_{target}_{name}.pkl")

            ev = va[sub_fit[va]]                           # 채점은 타깃 있는 행만
            s, one_m, f = group_score(y.iloc[ev].to_numpy(), pred[ev], cap)
            rows.append({"target": target, "fold": name, "n_fit": len(tr_fit),
                         "n_val": len(ev), "n_trees": n_used,
                         "score": s, "one_minus_nmae": one_m, "ficr": f})
            if verbose:
                print(f"   ├─ {name}: score={s:.4f}  1-NMAE={one_m:.4f}  FICR={f:.4f}  "
                      f"(fit {len(tr_fit):,} / trees {n_used})")

        if predict_all:
            oof[target] = pred
        else:
            oof.loc[fit_mask, target] = pred

    return oof, pd.DataFrame(rows), models


# ----------------------------------------------------------------- 중첩 평가
def nested_postprocess_score(df, answer_df, oof, targets, optimize_fn, apply_fn,
                             scheme="loyo", verbose=True, **opt_kwargs):
    """
    후처리 파라미터를 '평가할 연도를 뺀' OOF 에서 fit 하고 그 연도에 적용한다.
    기존 코드처럼 같은 OOF 에 fit 하고 같은 OOF 로 평가하면 개선폭이 과대추정된다.
    """
    df = add_time_keys(df)
    years = df["_year"].to_numpy()
    nested = oof.copy()

    for y in sorted(pd.unique(years)):
        te = years == y
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        params = optimize_fn(answer_df.loc[tr], oof.loc[tr], **opt_kwargs)
        nested.loc[te, targets] = apply_fn(oof.loc[te], params)[targets].to_numpy()

    raw = total_score(answer_df, oof, targets)
    nst = total_score(answer_df, nested, targets)
    if verbose:
        print(f"\n📐 [중첩 후처리 검증]")
        print(f"   raw OOF          : {raw[0]:.4f}  (1-NMAE {raw[1]:.4f} / FICR {raw[2]:.4f})")
        print(f"   nested 후처리    : {nst[0]:.4f}  (1-NMAE {nst[1]:.4f} / FICR {nst[2]:.4f})   "
              f"[{nst[0]-raw[0]:+.4f}]  <- 이게 정직한 개선폭")
    return nested, raw, nst


def nested_ensemble_weights(df, answer_df, oof_dict, targets, verbose=True):
    """앙상블 가중치도 동일하게 중첩. 평가 연도를 뺀 OOF 로 가중치를 구해 그 연도에 적용."""
    from scipy.optimize import minimize
    df = add_time_keys(df)
    years = df["_year"].to_numpy()
    names = list(oof_dict.keys())
    k = len(names)
    blended = oof_dict[names[0]].copy()
    per_year_w = {}

    for y in sorted(pd.unique(years)):
        te, tr = years == y, years != y
        if tr.sum() == 0 or te.sum() == 0:
            continue

        def loss(w):
            w = np.abs(w); w = w / w.sum()
            ens = answer_df.loc[tr, targets].copy()
            for c in targets:
                ens[c] = sum(w[i] * oof_dict[n].loc[tr, c].to_numpy() for i, n in enumerate(names))
            return -total_score(answer_df.loc[tr], ens, targets)[0]

        r = minimize(loss, np.full(k, 1.0 / k), method="Nelder-Mead",
                     options={"maxiter": 2000, "fatol": 1e-9})
        w = np.abs(r.x); w = w / w.sum()
        per_year_w[y] = dict(zip(names, np.round(w, 4)))
        for c in targets:
            blended.loc[te, c] = sum(w[i] * oof_dict[n].loc[te, c].to_numpy() for i, n in enumerate(names))

    if verbose:
        print("\n📐 [중첩 앙상블 가중치] 연도별로 뽑힌 가중치 (연도마다 크게 흔들리면 신뢰 불가)")
        for y, w in per_year_w.items():
            print(f"   {y}: {w}")
        for n in names:
            s = total_score(answer_df, oof_dict[n], targets)[0]
            print(f"   단일 {n:>10}: {s:.4f}")
        print(f"   중첩 앙상블      : {total_score(answer_df, blended, targets)[0]:.4f}")
    return blended, per_year_w


# ----------------------------------------------------------------- 오차막대 / 유의성
def score_by_year(df, answer_df, pred_df, targets=None, verbose=True, label="", min_rows=200):
    """
    연도별로 점수를 따로 낸다. 테스트가 '한 해 통째'이므로 연도 1개가 곧 1회 시행이고,
    그 편차가 이 대회에서 관측 가능한 노이즈 바닥이다.
    """
    targets = targets or TARGET_COLS
    df = add_time_keys(df)
    years = df["_year"].to_numpy()
    per = {}
    for y in sorted(pd.unique(years)):
        m = years == y
        if m.sum() < min_rows:                 # 꼬리처럼 몇 행만 걸친 연도는 제외
            continue
        try:
            per[int(y)] = total_score(answer_df.loc[m], pred_df.loc[m], targets)[0]
        except Exception:
            continue
    v = np.array(list(per.values()), dtype=float)
    v = v[~np.isnan(v)]
    if verbose:
        det = "  ".join(f"{k}:{s:.4f}" for k, s in per.items())
        print(f"   {label:24s} 평균 {v.mean():.4f}  연도별[{det}]  (표준편차 {v.std(ddof=1):.4f})")
    return per, float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan


def is_difference_real(df, answer_df, pred_a, pred_b, targets=None,
                       name_a="A", name_b="B", verbose=True, min_rows=200):
    """
    두 후보를 '연도별 짝지어(paired)' 비교한다. 연도 효과가 상쇄되므로 평균끼리
    비교하는 것보다 훨씬 민감하다.

    판정: 모든 연도에서 부호가 같고, |평균차| > 연도별 차이의 표준편차 이면 '신호'.
          아니면 노이즈 -> 그 실험은 제출 1회를 쓸 가치가 없다.
    """
    targets = targets or TARGET_COLS
    df = add_time_keys(df)
    years = df["_year"].to_numpy()
    diffs = {}
    for y in sorted(pd.unique(years)):
        m = years == y
        if m.sum() < min_rows:
            continue
        try:
            sa = total_score(answer_df.loc[m], pred_a.loc[m], targets)[0]
            sb = total_score(answer_df.loc[m], pred_b.loc[m], targets)[0]
        except Exception:
            continue
        if not (np.isnan(sa) or np.isnan(sb)):
            diffs[int(y)] = sa - sb
    d = np.array(list(diffs.values()), dtype=float)
    if len(d) < 2:
        if verbose:
            print("   연도가 2개 미만이라 유의성 판정 불가")
        return None
    same_sign = bool(np.all(d > 0) or np.all(d < 0))
    real = same_sign and abs(d.mean()) > d.std(ddof=1)
    if verbose:
        det = "  ".join(f"{k}:{v:+.4f}" for k, v in diffs.items())
        print(f"   {name_a} - {name_b}: 평균 {d.mean():+.4f}  연도별[{det}]")
        print(f"   판정: {'✅ 신호로 볼 만함' if real else '⚠️  노이즈 범위 — 제출 낭비 말 것'}"
              f"  (부호일치={same_sign}, |평균|={abs(d.mean()):.4f} vs 표준편차={d.std(ddof=1):.4f})")
    return {"diffs": diffs, "mean": float(d.mean()), "std": float(d.std(ddof=1)), "real": real}