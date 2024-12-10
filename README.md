# Yario Grid

이 프로젝트는 Super Mario Bros. 게임 환경에서 AI를 테스트하고 실험하기 위한 플랫폼입니다.  
[Chrispresso/SuperMarioBros-AI](https://github.com/Chrispresso/SuperMarioBros-AI)의 코드를 참고하여 제작되었습니다.

---

## NES 파일
- [NES 파일 다운로드](https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html)
- [NES 파일 사용에 대한 약관 확인](https://wowroms.com/en/disclaimer)

---

## 설치 및 실행

### 1. 가상환경 생성 및 실행
Python 3.8.19 버전을 사용하는 가상환경을 생성합니다.

```bash
conda create -n yario python=3.8.19
conda activate yario
```

### 2. 코드 클론
GitHub 저장소를 클론합니다.

```bash
git clone https://github.com/f56e751/yario_grid.git
```

### 3. 라이브러리 설치
클론한 폴더로 이동하여 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 4. YOLO 설치
YOLO를 설치하려면 아래 명령어를 사용합니다.  
[YOLO 설치 가이드](https://docs.ultralytics.com/guides/conda-quickstart/#setting-up-a-conda-environment)

```bash
conda install -c conda-forge ultralytics
```

### 5. PyTorch 설치
PyTorch를 GPU 또는 CPU 환경에 맞게 설치합니다.  
[PyTorch 설치 페이지](https://pytorch.org/get-started/locally/)  
구버전 PyTorch 설치는 [여기](https://pytorch.kr/get-started/previous-versions/)를 참고하세요.

---

## NES 파일 Retro에 등록

Super Mario Bros. NES 파일을 Retro 라이브러리에 등록합니다.

```bash
python -m retro.import "./Super Mario Bros. (World)"
```

---

## 실행 방법

`Main.py`를 실행하여 게임을 시작합니다.

```python
if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication 객체 생성
    main = Main(human_mode=False)
    main.run()
```

### 실행 모드
- `human_mode=False` → AI 플레이 모드
- `human_mode=True` → 사용자 플레이 모드

### 조작 키
- **점프**: `Space`
- **달리기**: `Shift`
- **이동**: 방향키

---

## YOLO 라벨 정의

게임 내 객체에 대한 YOLO 라벨은 아래와 같습니다:

```python
class YoloLabel(Enum):
    Mario_small = 0
    Mario_big = 1
    Mario_fire = 2
    Enemy = 3

    Mushroom = 4
    Flower = 5
    Star = 6
    LifeUp = 7

    Ground = 8
    Top_Pipe1 = 9
    Top_Pipe2 = 10
    Bottom_Pipe1 = 11
    Bottom_Pipe2 = 12
    Pipe_Horizontal = 13

    Flagpole_Top = 14
    Flagpole = 15
    Coin_Block = 16
    Coin_Block_End = 17
    Coin = 18
    Breakable_Block = 19
```

---

## 참고
- [Chrispresso/SuperMarioBros-AI](https://github.com/Chrispresso/SuperMarioBros-AI)
🚀
