import pygame
import retro
import numpy as np

# 게임 환경 설정
env = retro.make(game='SuperMarioBros-Nes')
env.reset()

# pygame 초기화 및 화면 설정
pygame.init()
screen = pygame.display.set_mode((256, 240))  # NES 해상도

running = True
clock = pygame.time.Clock()

while running:
    # pygame 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 버튼 상태를 0으로 초기화
    buttons = np.array([0] * 9)

    # 키보드 입력 받기
    keys = pygame.key.get_pressed()

    # Button Mappings:
    # 0: B
    # 1: None
    # 2: SELECT
    # 3: START
    # 5: DOWN
    # 6: LEFT
    # 7: RIGHT
    # 8: A
    
    if keys[pygame.K_RIGHT]:
        buttons[7] = 1  # 오른쪽 이동
        # if keys[pygame.K_SPACE]:
        #     buttons[0] = 1  # 오른쪽 이동 + B (빠르게 이동)
        # if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        #     buttons[8] = 1  # 오른쪽 이동 + A (점프)

    elif keys[pygame.K_LEFT]:
        buttons[6] = 1  # 왼쪽 이동
        # if keys[pygame.K_SPACE]:
        #     buttons[0] = 1  # 왼쪽 이동 + B (빠르게 이동)
        # if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        #     buttons[8] = 1  # 왼쪽 이동 + A (점프)

    elif keys[pygame.K_DOWN]:
        buttons[5] = 1  # 아래 방향키 (구부리기)
        
    if keys[pygame.K_SPACE]:
        buttons[8] = 1  # 단독 A (점프)
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        buttons[0] = 1  # 단독 B (빠르게 이동)

    # 환경에 액션 적용
    _obs, _rew, done, _info = env.step(buttons)
    if done:
        env.reset()
    
    # 게임 화면 가져오기 및 표시
    screen.blit(pygame.surfarray.make_surface(env.render(mode='rgb_array').swapaxes(0, 1)), (0, 0))
    pygame.display.flip()
    clock.tick(30)  # FPS를 30으로 제한

# 환경과 pygame 종료
pygame.quit()
env.close()
