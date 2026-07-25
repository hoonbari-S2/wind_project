import os
from datetime import datetime
import pandas as pd

# 로컬 하드웨어 고정 환경 정보
DEFAULT_ENV_SPEC = "CPU: AMD Ryzen 9 7950X | GPU: RTX 4080 SUPER | RAM: DDR5 64GB"


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
    log_path: str = "experiment_log.xlsx"
):
    """
    실험 결과를 experiment_log.xlsx 엑셀 파일에 자동으로 기록하는 함수
    """
    # config에서 학습 장치(CPU / GPU) 자동 추정
    if device is None:
        model_params = config.get("model_params", {})
        dev_param = str(model_params.get("device", model_params.get("tree_method", "cpu"))).lower()
        device = "GPU" if "cuda" in dev_param or "gpu" in dev_param else "CPU"

    # 엑셀에 저장할 행 데이터 정의
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Version": config.get("version", "unknown"),
        "Val_Total Score": round(total_score, 4),
        "Val_1 - NMAE": round(one_minus_nmae, 4),
        "Val_FICR": round(ficr, 4),
        "Execution Time (sec)": round(execution_time_sec, 2) if execution_time_sec is not None else None,
        "Model": config.get("model_type", "XGBoost"),
        "Device": device,  # <- CPU / GPU 명시
        "Seed": config.get("seed", 42),
        "Features": features_summary,
        "Environment Spec": env_spec,  # <- 하드웨어 사양 명시
        "Notes": notes
    }

    new_df = pd.DataFrame([log_data])

    if os.path.exists(log_path):
        try:
            existing_df = pd.read_excel(log_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            updated_df = new_df
    else:
        updated_df = new_df

    updated_df.to_excel(log_path, index=False, engine="openpyxl")
    print(f"📝 [Logger] 실험 결과가 '{log_path}' 파일에 성공적으로 기록되었습니다.")