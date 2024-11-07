https://github.com/Chrispresso/SuperMarioBros-AI 의 코드를 참고함


nes 파일
https://wowroms.com/en/disclaimer
https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html




가상환경 생성

    conda create -n yario python=3.8.19

가상환경 실행

    conda activate yario

클론

    git clone https://github.com/f56e751/yario_grid.git
    
라이브러리 설치 (클론한 폴더에서 실행)

    pip install -r requirements.txt

Yolo 설치
https://docs.ultralytics.com/guides/conda-quickstart/#setting-up-a-conda-environment

    conda install -c conda-forge ultralytics

nes 파일 retro에 등록

    python -m retro.import "./Super Mario Bros. (World)"



    
위 명령어 실행 후 mario_retro_showgrid 실행

점프: space
달리기: shift
이동: 방향키


class YoloLabel(Enum):
    
    Mario_small = 0
    Mario_big = 1
    Mario_fire = 2
    Enemy = 3


    Mushroom = 4
    Flower = 5
    Star = 6
    LifeUp = 7


    # Empty = 0x00
    Ground = 8
    Top_Pipe1 = 9
    Top_Pipe2 = 10
    Bottom_Pipe1 = 11
    Bottom_Pipe2 = 12
    Pipe_Horizontal = 13


    Flagpole_Top =  14
    Flagpole = 15
    Coin_Block = 16
    Coin_Block_End = 17
    Coin = 18

    Breakable_Block = 19
