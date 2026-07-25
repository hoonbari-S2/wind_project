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
COLUMNS = [
    "Timestamp", "Version",
    "Validation", "Target", "ES Mode", "Post Mode",     # ← 신규: 비교 가능성 판단용
    "Raw OOF", "Val_Total Score", "Val_1 - NMAE", "Val_FICR", "Year Spread",
    "Public LB", "Private LB",                          # ← 수동 기입
    "Execution Time (sec)", "Model", "Device", "Seed", "N Features",
    "Features", "Environment Spec", "Notes",
]

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
        "Raw OOF": round(raw_oof, 4) if raw_oof is not None else None,
        "Val_Total Score": round(total_score, 4),
        "Val_1 - NMAE": round(one_minus_nmae, 4),
        "Val_FICR": round(ficr, 4),
        "Year Spread": round(year_spread, 4) if year_spread is not None else None,
        "Public LB": None,          # 제출 후 직접 기입
        "Private LB": None,
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
            df = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            df = new_df
    else:
        df = new_df

    # 열 순서 고정 (과거 기록에 없던 열은 NaN 으로 남음)
    ordered = [c for c in COLUMNS if c in df.columns] + [c for c in df.columns if c not in COLUMNS]
    df = df[ordered]
    df.to_excel(log_path, index=False, engine="openpyxl")

    print(f"📝 [Logger] '{log_path}' 기록 완료  "
          f"(검증={validation} / 타깃={row['Target']} / 오차막대±{row['Year Spread']})")
    if validation == "loyo":
        print("   ⚠️ LOYO 점수는 월 GroupKFold 기록(v1~v10)과 절대값 비교 불가. "
              "같은 Validation 끼리만 비교할 것.")