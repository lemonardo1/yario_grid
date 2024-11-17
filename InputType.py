import numpy as np
import pygame

from network import PPOAgent


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
    def __init__(self, agent: PPOAgent):
        self.action = np.array([0] * 9)  # 액션 초기화
        self.agent = agent
        # 0부터 순서대로 
        # null, 아래(숙이기), 좌, 우, a=jump, b=달리기 or 공격, 우 + 점프, 좌 + 점프, 우 + 공격, 좌 + 공격, 우 + 공격 + 점프, 좌 + 공격 + 점프
        # # Keys correspond with          B, NULL, SELECT, START, U, D, L, R, A
        # # index                         0  1     2       3      4  5  6  7  8
        self.action_map = { 0: np.array( [0, 1,    0,      0,     0, 0, 0, 0, 0], np.int8),
                            1: np.array( [0, 0,    0,      0,     0, 1, 0, 0, 0], np.int8),
                            2: np.array( [0, 0,    0,      0,     0, 0, 1, 0, 0], np.int8),
                            3: np.array( [0, 0,    0,      0,     0, 0, 0, 1, 0], np.int8),
                            4: np.array( [0, 0,    0,      0,     0, 0, 0, 0, 1], np.int8),
                            5: np.array( [1, 0,    0,      0,     0, 0, 0, 0, 0], np.int8),
                            6: np.array( [0, 0,    0,      0,     0, 0, 0, 1, 1], np.int8),
                            7: np.array( [0, 0,    0,      0,     0, 0, 1, 0, 1], np.int8),
                            8: np.array( [1, 0,    0,      0,     0, 0, 0, 1, 0], np.int8),
                            9: np.array( [1, 0,    0,      0,     0, 0, 1, 0, 0], np.int8),
                           10: np.array( [1, 0,    0,      0,     0, 0, 0, 1, 1], np.int8),
                           11: np.array( [1, 0,    0,      0,     0, 0, 1, 0, 1], np.int8),
                           }


    def get_action(self, state):
        
        # action은 0 ~ num action - 1 의 범위의 정수
        action, action_tensor, log_prob, value = self.agent.select_action(state)

        self.action = self.action_map[action]
        return self.action
    
    def get_action_np(self, action):
        for key, value in self.action_map.items():
            if key == action:
                action_np = value
                return action_np

        raise ValueError(f"Action '{action}' not found in action map.")
    

    # 넘파이 형식을 받아 int 형식의 action을 반환
    def get_action_int(self, action):
        for key, value in self.action_map.items():
            if np.array_equal(value, action):
                return key

        raise ValueError(f"Action '{action}' not found in action map.")
    

    def get_null_action(self):
        action = np.array([0] * 9)
        action[1] = 1
        return action
    
    def get_jump_action(self):
        action = np.array([0] * 9)
        action[8] = 1
        return action