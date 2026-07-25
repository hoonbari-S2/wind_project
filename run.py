import argparse
import runpy
import sys
from pathlib import Path

# 프로젝트 최상위 경로를 파이썬 검색 경로에 보장
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="BARAM 2026 Pipeline Runner")
    parser.add_argument(
        "mode", 
        choices=["train", "inference"], 
        help="실행할 파이프라인 선택 ('train' 또는 'inference')"
    )
    args = parser.parse_args()

    if args.mode == "train":
        print("🚀 [Pipeline] main/train.py 학습 및 후처리 파이프라인을 실행합니다.\n")
        runpy.run_module("main.train", run_name="__main__")
    elif args.mode == "inference":
        print("🚀 [Pipeline] main/inference.py 추론 파이프라인을 실행합니다.\n")
        runpy.run_module("main.inference", run_name="__main__")

if __name__ == "__main__":
    main()