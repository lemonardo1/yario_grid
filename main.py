from Game import Game
from InputType import HumanInput, AgentInput
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from Visualizer import GameFrameVisualizer, GridWindow
from Yolo_Model import Yolo_Model
from PIL import Image
from network import PPOAgent
from Trainer import Trainer

import numpy as np
from PyQt5.QtWidgets import QApplication
import time
import pygame
import sys
import threading
import torch

# self.game.is_playable()
# self.game.is_dead()
# self.game.is_world_cleared()
# self.game.reset()
# 위 메서드를 ppo training에 적용


# TODO get_yolo_format_for_game로 타일 정보 받으면 좀 이상한데 다시 기존 방식으로도 다시 해보기
class Main:
    def __init__(self, human_mode=True, use_yolo = True, training = False, visualize = True, grid_visualize = False):
        pygame.init()
        self.fps = 60
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.clock = pygame.time.Clock()
        self.game = Game(self.x_pixel_num, self.y_pixel_num, visualize)
        # self.gameFrameVisualizer = GameFrameVisualizer()
        self.grid_window = GridWindow()
        self.grid_visualize = grid_visualize
        self.human_mode = human_mode
        self.use_yolo = use_yolo
        self.training = training
        
        self.yolo_model = None
        self.agent = None


        self.is_grid_shown = False
        
        if self.grid_visualize:
            self.grid_window.show()
            self.init_grid_window(self.grid_window)

        # null action으로 초기화함
        self.prev_action = np.array( [0, 1,    0,      0,     0, 0, 0, 0, 0], np.int8)
        self.action_dim = 12
        if self.use_yolo:
            self.yolo_model = Yolo_Model(self.x_pixel_num, self.y_pixel_num)
            # self.yolo_model.set_logging()
        
        if self.human_mode:
            self.input_device = HumanInput()
        else:
            self.init_agent_mode()


    def init_agent_mode(self):
        if not self.training:
            input_dim = (15 * 16 * 4 + 3) * 4 + 12 # (15 * 16 * 4 + 3(마리오 상태) ) * 4 + 12(action) : 3864
            # TODO 이전 action 12개 더하기 
            hidden_dims = [1024, 256]
            output_dim = 12                   # action 12개

            # TODO 여기서 기존에 저장한 파라미터 불러올 수 있게
            self.agent = PPOAgent(input_dim, hidden_dims, output_dim)
            self.input_device = AgentInput(self.agent)


            self.game_thread = threading.Thread(target=self.game.run)
            self.game_thread.start()
        else:
            trainer = Trainer(self.game, self.use_yolo)
            # trainer.train_test()
            trainer.train()

    def init_grid_window(self, grid_window):
        self.is_grid_shown = True
        self.grid_window = grid_window

    def deacvivate_visualize(self):
        return 

    def show_grid(self):
        if not self.is_grid_shown:
            return

        if self.use_yolo:
            # human mode의 경우 yolo를 키지 않고 작동하기 때문에 전처리과정이 필요
            if self.human_mode:
                yolo_input_img = self.game.get_yolo_input_img()
                mario_state = self.game.get_mario_state() # 0: small, 1: big,  => 2 : firey
                tensor = self.yolo_model.get_tensor(yolo_input_img, mario_state)
            
            tiles = self.yolo_model.get_grid_visualize_tile()
            self.grid_window.update_tiles(tiles)
            
        else:
            # ram data 사용
            tiles = self.game.get_tile_info()
            self.grid_window.update_tiles(tiles)



    def run(self):
        # training 모드이면 run 코드가 실행되지 않음
        if self.training:
            return
        
        running = True
        last_update_time = 0
        update_interval = 1000 / self.fps  # 업데이트 간격 (밀리초 단위)
        count = 1
        try:
            while running:
                count += 1
                current_time = pygame.time.get_ticks()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # print("main test class quit")
                        self.game.stop()
                        running = False
                if self.human_mode:
                    
                    action = self.input_device.get_action()
                    self.game.update_game_human_mode(action)
                    self.game.is_playable()
                    self.game.is_dead()
                    self.game.is_world_cleared()

                    keys = pygame.key.get_pressed()  
                    if keys[pygame.K_n]:
                        print('n pressed')
                        self.game.reset()
                else:
                    # 에이전트 모드일 때 게임 로직
                    # print('agent mode')
                    # if current_time - last_update_time < update_interval:
                    #     continue
                
                    yolo_input_img = self.game.get_yolo_input_img()
                    mario_state = self.game.get_mario_state() # 0: small, 1: big,  => 2 : firey

                    if self.use_yolo:
                        tensor = self.yolo_model.get_tensor(yolo_input_img, mario_state)
                    else:
                        tensor = self.game.get_tensor()
                        
                    # 누적 프레임만큼 쌓이지 않으면 tensor = None
                    # TODO 프레임 쉬는 동안에는 이전 액션을 취하도록
                    if tensor != None:
                        # print('tensor not none')
                        # tensor 합치기
                        prev_action_one_hot = torch.zeros(self.action_dim)
                        prev_action_int = self.input_device.get_action_int(self.prev_action)
                        prev_action_one_hot[prev_action_int] = 1
                        full_state = torch.cat([tensor, prev_action_one_hot])
                        action = self.input_device.get_action(full_state)
                        # print(action)
                        self.game.receive_action(action)
                        last_update_time = current_time
                        self.prev_action = action

                    else:
                        # print('tensor is none')
                        self.game.receive_action(self.prev_action)
                        last_update_time = current_time

                self.show_grid()

                self.clock.tick(self.fps) # 이 부분을 수정하면 fps 제한 없이 연산, 업데이트 를 반복하며 프레임 손실 없이 가능
        finally:
            # if not self.human_mode:
            if False:
                self.game_thread.join()  # 게임 스레드가 종료될 때까지 기다림



def yolo_test():
    yolo = Yolo_Model(256, 240)
    results = yolo.yolo_test('test_img.png')
    for result in results:
        boxes = result.boxes
        x1 = boxes.xyxy[:,0]
        x2 = boxes.xyxy[:,2]
        y1 = boxes.xyxy[:,1]
        y2 = boxes.xyxy[:,3]
        cls = boxes.cls

    print('end')

if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication 객체 생성

    # 주의사항!!!
    # visualize = True일 때 training 진행 시, 
    # 현재 게임창을 alt + tab으로 다른 창으로 전환하면 게임 창이 더이상 업데이트되지 않는 현상이 발생

    # human_mode=True, use_yolo = False, training = False, visualize = True, grid_visualize = True
    # 위 5개의 bool을 바꿔가면서 실행 가능
    # 1. training이 true이면 human_mode 이 true, false 일 때 상관없이 training이 진행됨
    # 2. use_yolo가 true이면  
    #      training = False, human_mode=False 일 때, yolo를 이용해 tensor를 생성
    #      training = True 일 때, yolo를 이용해 생성한 tensor를 training에 사용
    # 3. human_mode = True -> 사람 입력 가능,
    #    human_mode = False -> agent 입력  (아직 training된 파라미터를 불러오는 기능은 없음)
    # 4. visualize = True -> human_mode = True, False, Training 모드에서 게임창 시각화 가능
    #    visualize = Fasle -> 위의 모든 모드에서 시각화 제외 후 입력
    # 5. grid_visualize = True -> 그리드 형식으로 현재 게임이 나옴
    #    use_yolo가 true이면 yolo에서 인식된 결과를 출력
    #    use_yolo가 false이면 게임에서 직접 불러온 결과를 출력
    
    main = Main(human_mode=True, use_yolo = True, training = False, visualize = True, grid_visualize = True)

    main.run()

