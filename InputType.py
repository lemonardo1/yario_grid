import numpy as np
import pygame


class HumanInput():
    def __init__(self):
        self.action = np.array([0] * 9)  # 액션 초기화
    
    def get_action(self):
        keys = pygame.key.get_pressed()  
        self.action = np.array([0] * 9)


        # # Keys correspond with             B, NULL, SELECT, START, U, D, L, R, A
        # # index                            0  1     2       3      4  5  6  7  8
        # self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        # 키 입력에 따라 액션 설정
        if keys[pygame.K_RIGHT]:
            self.action[7] = 1
        if keys[pygame.K_LEFT]:
            self.action[6] = 1
        if keys[pygame.K_SPACE]:
            self.action[8] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            self.action[0] = 1
        if keys[pygame.K_DOWN]:
            self.action[5] = 1
        if keys[pygame.K_UP]:
            self.action[4] = 1

        return self.action


class AgentInput():
    def __init__(self):
        self.action = np.array([0] * 9)  # 액션 초기화
        self.frame_counter = 0

    def get_action(self):
        # 여기에서 에이전트 로직을 구현 (예시)
        # 매 호출 시마다 다른 액션을 반환할 수 있음
        self.action = np.array([0] * 9)
        self.frame_counter += 1
        # self.action = (self.frame_counter // 100) % 2 * np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
        self.action[7] = 1
        return self.action