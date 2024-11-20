import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque
import numpy as np
import os

from Game import Game
from network import PPOAgent
from Yolo_Model import Yolo_Model
from InputType import AgentInput

# 하이퍼파라미터
NUM_EPISODES = 1000
MAX_STEPS = 100000#30* 401 # 한 프레임30, 총 400초
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_INTERVAL = 500  # 업데이트 주기
BATCH_SIZE = 64
CLIP_EPSILON = 0.15  # PPO 클리핑 epsilon 값
LR = 3e-3  # 학습률

class Trainer():
    def __init__(self, game: Game, use_yolo = False):
        # 에이전트 및 모델 초기화
        input_dim = 3864  # YOLO state + Mario state + 이전 행동
        self.action_dim = 12  # 12차원 행동 공간
        hidden_dims = [1024, 256]  # 네트워크 hidden layer 크기
        self.agent = PPOAgent(input_dim = input_dim, hidden_dims = hidden_dims, output_dim = self.action_dim)  # PPO 에이전트
        self.AgentInput = AgentInput(self.agent)
        self.yolo_model = Yolo_Model(x_pixel_num=256, y_pixel_num=240)  # YOLO 모델
        self.game = game  # 게임 환경

        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.agent.parameters(), lr=LR)
        self.use_yolo = use_yolo


        # 0부터 순서대로 
        # null, 아래(숙이기), 좌, 우, a=jump, b=달리기 or 공격, 우 + 점프, 좌 + 점프, 우 + 공격, 좌 + 공격, 우 + 공격 + 점프, 좌 + 공격 + 점프
        # 0 ~ 11의 정수
        self.prev_action = 0 

        ### 저장
        self.model_dir = './models'  # 모델을 저장할 디렉터리
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, episode):
        """ 모델의 파라미터를 저장합니다. """
        current_time = time.time()
        filename = os.path.join(self.model_dir, f"ppo_agent_episode_{episode}_{current_time}.pth")
        torch.save(self.agent.state_dict(), filename)
        print(f"Model saved to {filename} at episode {episode}")

    def get_tensor(self):
        if self.use_yolo:
            mario_state = self.game.get_mario_state()  # 4프레임 동안의 마리오 상태 (12차원 벡터)
            # YOLO 모델을 사용하여 상태 추출
            yolo_input_img = self.game.get_yolo_input_img()
            tensor_state = self.yolo_model.get_tensor(yolo_input_img, mario_state)
            return tensor_state
        else:
            tensor_state = self.game.get_tensor()
            # print(tensor_state)
            return tensor_state

    def train_test(self):
        action = np.array([0] * 9)
        action[8] = 1
        for i in range(10000000):
            reward, done, _ = self.game.step(action)
            time.sleep(1/60)


    def train(self):
        for episode in range(NUM_EPISODES):
            state = None  # 초기 상태
            states = []
            rewards, log_probs, values, masks, actions = [], [], [], [], []
            old_log_probs, old_values = [], []  # 이전 log_prob와 value를 저장
            print(f"episode {episode} start")
            start_time = time.time()
            actual_steps = 0
            for step in range(MAX_STEPS):
                current_time = time.time()

                   
                tensor_state = self.get_tensor()
                # 누적 프레임이 충분하지 않으면 이전 action을 입력
                if tensor_state is None:
                    action_np = self.AgentInput.get_action_np(self.prev_action)
                    reward, done, _ = self.game.step(action_np)
                    continue
                actual_steps += 1
                # 이전 action을 one-hot 벡터로 결합
                # prev_action = self.game.get_prev_action_index()  # 12차원 벡터 (이전 행동)
                prev_action_one_hot = torch.zeros(self.action_dim)
                prev_action_one_hot[self.prev_action] = 1
                
                # # 마리오 상태 12차원 벡터 (이미 4프레임 동안의 정보가 제공됨)
                # mario_state_tensor = torch.tensor(mario_state, dtype=torch.float32)

                # 입력 벡터를 결합
                # full_state = torch.cat([tensor_state, mario_state_tensor, prev_action_one_hot])
                full_state = torch.cat([tensor_state, prev_action_one_hot])

                # 에이전트 행동 선택
                action_int, action_tensor, log_prob, value = self.agent.select_action(full_state)
                self.prev_action = action_int
                action_np = self.AgentInput.get_action_np(action_int) # action = np.array([0] * 9)
                
                action_one_hot = torch.zeros(self.action_dim)
                action_one_hot[action_int] = 1
                # 보상 및 게임 정보 업데이트
                reward, done, _ = self.game.step(action_np)  # step() 메소드에서 보상과 종료 여부 받기
                
                
                if actual_steps % 100 == 0:
                    print(f"prev action: {self.prev_action}")
                    print(f"current_step: {step}")
                    print(f"elapsed time: {current_time - start_time}")
                    print(f"reward: {reward}")
                    print(f"log_prob: {log_prob}")
                
                
                # print(f"reward: {reward}")
                # print(f"action: {action_int}")
                # log_prob = 
                states.append(full_state)  # 현재 상태를 states 리스트에 추가
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                masks.append(1 - done)
                actions.append(action_one_hot)
                old_log_probs.append(log_prob.detach())  # 이전 log_prob 저장
                old_values.append(value.detach())  # 이전 value 저장
                # 에피소드 종료 조건
                if done:
                    break

                # 업데이트 주기마다 에이전트를 학습
                if actual_steps % UPDATE_INTERVAL == 0:
                    print('update interval')
                    if not log_probs:  # log_probs 리스트가 비어 있으면 업데이트를 스킵
                        print("Skipping update: 'log_probs' list is empty.")
                        continue

                    # 모든 리스트의 길이가 동일한지 확인
                    if not all(len(lst) == len(log_probs) for lst in [rewards, values, masks, old_log_probs, old_values]):
                        print("Skipping update: Not all lists are of equal length.")
                        continue

                    # 리스트 내 요소의 차원 확인
                    if any(lp.dim() == 0 for lp in log_probs):  # log_probs의 각 요소가 0차원인지 확인
                        print("Skipping update: 'log_probs' contains zero-dimensional tensors.")
                        continue
                    returns, advantages = self.compute_gae(rewards, values, masks)
                    
                    # PPO 업데이트: 클리핑 기법을 사용하여 정책 업데이트
                    self.agent.update(states, actions, returns, advantages, log_probs, old_log_probs, old_values, self.optimizer, BATCH_SIZE, CLIP_EPSILON)
                    print("save_model")
                    # self.save_model(episode)
                    # 매 업데이트 후 로그 초기화
                    rewards, log_probs, values, masks, actions = [], [], [], [], []
                    old_log_probs, old_values = [], []
                    states = []

            print(f"Episode {episode + 1}/{NUM_EPISODES} completed.")
            
    def compute_gae(self, rewards, values, masks, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        # GAE advantage 계산
        values = values + [0]  # 마지막 상태 값 추가
        returns, advantages = [], []
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * gae_lambda * masks[i] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return returns, advantages



