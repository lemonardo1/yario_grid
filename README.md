https://github.com/Chrispresso/SuperMarioBros-AI 의 코드를 참고함


https://wowroms.com/en/disclaimer
https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html

위에서 nes 파일 다운해서 현재 폴더에 압축풀기

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
