"""
src/logger.py — 실험 로그 자동 기록
==============================================================================
v11 개정: 검증 방식 / 타깃 종류 / 오차막대 열 추가

왜 필요한가
  검증 방식이 다르면 OOF 절대값을 비교할 수 없다. 월 GroupKFold 는 각 폴드가
  3년에 흩어진 80% 로 학습하지만 LOYO 는 2년치(67%)로 미지의 1년을 맞힌다.
  같은 모델이라도 LOYO 쪽이 구조적으로 낮게 나온다.
  열에 안 남기면 몇 주 뒤에 v6(0.6087) 와 v11(0.61) 을 나란히 놓고
  "왜 안 올랐지" 하게 된다.

  Year Spread(연도별 표준편차)도 같은 이유다. 두 버전의 차이가 이 값보다
  작으면 노이즈다. 점수만 적어두면 그 판단을 나중에 할 수 없다.
==============================================================================
"""
import os
from datetime import datetime

import pandas as pd

DEFAULT_ENV_SPEC = "CPU: AMD Ryzen 9 7950X | GPU: RTX 4080 SUPER | RAM: DDR5 64GB"

# 엑셀 열 순서 (스키마가 늘어나도 순서를 고정)
# v14 개정: LB 를 점수/NMAE/FICR 3열로 쪼개고 맨 뒤로 옮긴다.
#   LB 총점만 적어두면 "왜 떨어졌나" 를 나중에 분해할 수 없다. v14 가 정확히 그 경우였다
#   (총점 -0.0120 인데 내역은 NMAE -0.0051 / FICR -0.0189 로 FICR 이 3.7배 크게 깨짐).
#   그 분해가 있어야 '밴드 문제인가 정확도 문제인가' 를 제출 없이 판별할 수 있다.
COLUMNS = [
    "Timestamp", "Version",
    "Validation", "Target", "ES Mode", "Post Mode",     # 비교 가능성 판단용
    "Objective", "Sample Weight",                       # ← v14 신규: 학습 문제 정의
    "Raw OOF", "Val_Total Score", "Val_1 - NMAE", "Val_FICR", "Year Spread",
    "Execution Time (sec)", "Model", "Device", "Seed", "N Features",
    "Features", "Environment Spec", "Notes",
    "Public_LB_Score", "Public_LB_1-nMAE", "Public_LB_FiCR",   # ← 제출 후 수동 기입
    "private_score",                                           # ← 대회 종료 후
]

# 구 스키마 -> 신 스키마 (과거 기록을 잃지 않고 옮긴다)
RENAME_LEGACY = {"Public LB": "Public_LB_Score", "Private LB": "private_score"}


def _repair(df):
    """
    엑셀 수기 편집이 만드는 두 가지 파손을 복구한다.

      (1) 완전 공백 행 — 그냥 버린다.
      (2) Notes 흘러넘침 — 여러 줄짜리 Notes 를 셀에 붙여넣으면 엑셀이 줄바꿈마다
          새 행을 만든다. Version 이 비어 있고 Notes 만 있는 행이 그것이다.
          버리면 기록이 사라지므로, 바로 위 정상 행의 Notes 에 도로 이어붙인다.

    이걸 안 하면 새 기록이 공백 행 아래로 밀려서 로그 정렬이 깨진다.
    """
    if "Version" not in df.columns:
        return df.dropna(how="all")
    keep, spilled = [], 0
    for _, r in df.iterrows():
        has_ver = pd.notna(r.get("Version")) and str(r.get("Version")).strip() != ""
        others = r.drop(labels=["Notes"], errors="ignore")
        if has_ver:
            keep.append(r.copy())
        elif others.isna().all() and pd.notna(r.get("Notes")) and keep:
            prev = keep[-1]
            prev["Notes"] = f"{prev.get('Notes') or ''}\n{r['Notes']}".strip()
            spilled += 1
        # else: 완전 공백 -> 버림
    if spilled:
        print(f"🩹 [Logger] 엑셀에서 쪼개진 Notes {spilled}줄을 원래 행에 되붙였다.")
    return pd.DataFrame(keep).reset_index(drop=True) if keep else df.dropna(how="all")

VALIDATION_NOTE = {
    "loyo": "LOYO(연 단위) — 테스트(2025 통째)와 동일 구조. month_group 과 절대값 비교 불가",
    "month_group": "월 GroupKFold — 폴드마다 3년에 흩어진 80% 학습. loyo 보다 구조적으로 높게 나옴",
    "block_month": "연속 월 블록 K-fold",
    "holdout2024": "2024 단일 홀드아웃",
}


def log_experiment(
    config: dict,
    total_score: float,
    one_minus_nmae: float,
    ficr: float,
    execution_time_sec: float = None,
    device: str = None,
    env_spec: str = DEFAULT_ENV_SPEC,
    features_summary: str = "",
    notes: str = "",
    log_path: str = "experiment_log.xlsx",
    # --- v11 신규 (전부 선택 인자라 기존 호출도 그대로 동작) ---
    validation: str = None,
    target_kind: str = None,
    es_mode: str = None,
    post_mode: str = None,
    raw_oof: float = None,
    year_spread: float = None,
    n_features: int = None,
    # --- v14 신규 ---
    objective: str = None,
    sample_weight: str = None,
):
    """실험 결과 1행을 experiment_log.xlsx 에 append."""
    if device is None:
        mp = config.get("model_params", {})
        dev = str(mp.get("device", mp.get("tree_method", "cpu"))).lower()
        device = "GPU" if ("cuda" in dev or "gpu" in dev) else "CPU"

    # yaml 에 적어둔 요약을 우선 사용하고, 없을 때만 인자값
    features_summary = config.get("features_summary") or features_summary
    notes = config.get("notes") or notes

    validation = validation or config.get("validation_scheme", "unknown")
    if validation in VALIDATION_NOTE and VALIDATION_NOTE[validation] not in (notes or ""):
        notes = f"{notes}  ⟨검증: {VALIDATION_NOTE[validation]}⟩".strip()

    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Version": config.get("version", "unknown"),
        "Validation": validation,
        "Target": target_kind or "raw_label",
        "ES Mode": es_mode or "-",
        "Post Mode": post_mode or "-",
        "Objective": objective or config.get("model_params", {}).get("objective", "-"),
        "Sample Weight": sample_weight or "none",
        "Raw OOF": round(raw_oof, 4) if raw_oof is not None else None,
        "Val_Total Score": round(total_score, 4),
        "Val_1 - NMAE": round(one_minus_nmae, 4),
        "Val_FICR": round(ficr, 4),
        "Year Spread": round(year_spread, 4) if year_spread is not None else None,
        "Public_LB_Score": None,    # 제출 후 직접 기입
        "Public_LB_1-nMAE": None,
        "Public_LB_FiCR": None,
        "private_score": None,      # 대회 종료 후
        "Execution Time (sec)": round(execution_time_sec, 2) if execution_time_sec else None,
        "Model": config.get("model_type", "XGBoost"),
        "Device": device,
        "Seed": config.get("seed", 42),
        "N Features": n_features,
        "Features": features_summary,
        "Environment Spec": env_spec,
        "Notes": notes,
    }

    new_df = pd.DataFrame([row])
    if os.path.exists(log_path):
        try:
            old = pd.read_excel(log_path)
            old = _repair(old.rename(columns=RENAME_LEGACY))     # 구 스키마 흡수 + 파손 복구
            df = pd.concat([old, new_df], ignore_index=True)
        except Exception as e:
            print(f"⚠️ [Logger] 기존 로그를 못 읽었다({type(e).__name__}). 새 파일로 시작한다.")
            df = new_df
    else:
        df = new_df

    # 열 순서 고정. 새 스키마 열이 과거 기록에 없으면 만들어 준다.
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    ordered = COLUMNS + [c for c in df.columns if c not in COLUMNS]
    df = df[ordered]
    df.to_excel(log_path, index=False, engine="openpyxl")

    print(f"📝 [Logger] '{log_path}' 기록 완료  "
          f"(검증={validation} / 타깃={row['Target']} / 오차막대±{row['Year Spread']})")
    if validation == "loyo":
        print("   ⚠️ LOYO 점수는 월 GroupKFold 기록(v1~v10)과 절대값 비교 불가. "
              "같은 Validation 끼리만 비교할 것.")