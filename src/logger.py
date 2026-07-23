import os
from datetime import datetime
import pandas as pd

def log_experiment(
    config: dict,
    total_score: float,
    one_minus_nmae: float,
    ficr: float,
    features_summary: str = "",
    notes: str = "",
    log_path: str = "experiment_log.xlsx"
):
    """
    실험 결과를 experiment_log.xlsx 엑셀 파일에 자동으로 기록하는 함수
    """
    # 엑셀에 저장할 행 데이터 정의
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Version": config.get("version", "unknown"),
        "Val_Total Score": round(total_score, 4),
        "Val_1 - NMAE": round(one_minus_nmae, 4),
        "Val_FICR": round(ficr, 4),
        "Model": config.get("model_type", "RandomForest"),
        "Seed": config.get("seed", 42),
        "Features": features_summary,
        "Notes": notes
    }

    new_df = pd.DataFrame([log_data])

    # 기존 엑셀 파일이 존재하면 아래에 덧붙이기(Append), 없으면 새로 생성
    if os.path.exists(log_path):
        try:
            existing_df = pd.read_excel(log_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            updated_df = new_df
    else:
        updated_df = new_df

    # openpyxl 엔진을 사용하여 엑셀 파일로 저장
    updated_df.to_excel(log_path, index=False, engine="openpyxl")
    print(f"📝 [Logger] 실험 결과가 '{log_path}' 파일에 자동 저장되었습니다.")