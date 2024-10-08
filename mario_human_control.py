import pygame
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT  # COMPLEX_MOVEMENT 사용
from utils import SMB, EnemyType, StaticTileType, DynamicTileType, ColorMap  # utils 파일에서 필요한 클래스 및 열거형 가져오기

# pygame 초기화
pygame.init()
screen = pygame.display.set_mode((640, 480))

# 환경을 생성하고 JoypadSpace로 감쌉니다.
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = gym_super_mario_bros.make('SuperMarioBros-v3')

env = JoypadSpace(env, COMPLEX_MOVEMENT)  # SIMPLE_MOVEMENT 대신 COMPLEX_MOVEMENT 사용

done = True
clock = pygame.time.Clock()

# 타일 크기 및 화면 오프셋 설정
tile_width, tile_height = 16, 16  # 타일의 크기
x_offset = 150  # 화면에서 타일을 그리기 시작할 x 좌표

while True:
    # Pygame 이벤트 큐에서 이벤트를 가져옵니다.
    for event in pygame.event.get():
        if event.type is pygame.QUIT:
            env.close()
            pygame.quit()
            exit()
    
    keys = pygame.key.get_pressed()
    
    action = 0  # 기본 동작 (아무것도 안 함)
    if keys[pygame.K_RIGHT]:
        action = 1  # 오른쪽 이동
        if keys[pygame.K_SPACE]:
            action = 2  # 오른쪽 이동 + B (빠르게 이동)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action = 3  # 오른쪽 이동 + A (점프)
        if keys[pygame.K_SPACE] and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
            action = 4  # 오른쪽 이동 + B + A (빠르게 이동 + 점프)
    elif keys[pygame.K_LEFT]:
        action = 6  # 왼쪽 이동
        if keys[pygame.K_SPACE]:
            action = 7  # 왼쪽 이동 + B (빠르게 이동)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action = 8  # 왼쪽 이동 + A (점프)
        if keys[pygame.K_SPACE] and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
            action = 9  # 왼쪽 이동 + B + A (빠르게 이동 + 점프)
    elif keys[pygame.K_SPACE]:
        action = 5  # 단독 B (빠르게 이동)
    elif keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        action = 11  # 단독 A (점프)
    elif keys[pygame.K_DOWN]:
        action = 10  # 아래 방향키 (구부리기)

    if done:
        state = env.reset()
    
    state, reward, done, info = env.step(action)  # 수정된 조작 입력
    env.render()


    pygame.display.flip()

    clock.tick(30)  # FPS를 30으로 제한

pygame.quit()
