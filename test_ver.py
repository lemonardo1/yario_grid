import asyncio
import sys

# 현재 Python 버전 출력
print(f"Current Python version: {sys.version}")

import retro

import ultralytics
import torch

print(f"PyTorch version: {torch.__version__}")

import gym
print(gym.__version__)



import sys
import os

# 가상환경 이름 확인
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # 가상환경이 활성화된 경우
    venv_path = sys.prefix
    venv_name = os.path.basename(venv_path)
    print(f"현재 활성화된 가상환경: {venv_name}")
else:
    # 가상환경이 활성화되지 않은 경우
    print("현재 가상환경이 활성화되지 않았습니다.")

# 추가로 Python 실행 경로 출력
print(f"Python 실행 경로: {sys.executable}")