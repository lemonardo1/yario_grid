from Game import Game
from InputType import HumanInput, AgentInput
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from Visualizer import GameFrameVisualizer, GridWindow
from Yolo_Model import Yolo_Model
from PIL import Image
from network import PPOAgent

import numpy as np
from PyQt5.QtWidgets import QApplication
import time
import pygame
import sys
import threading

# self.game.is_playable()
# self.game.is_dead()
# self.game.is_world_cleared()
# self.game.reset()
# 위 메서드를 ppo training에 적용


# TODO get_yolo_format_for_game로 타일 정보 받으면 좀 이상한데 다시 기존 방식으로도 다시 해보기
class Main:
    def __init__(self, human_mode=True, use_yolo = True):
        pygame.init()
        self.fps = 60
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.clock = pygame.time.Clock()
        self.game = Game(self.x_pixel_num, self.y_pixel_num)
        self.gameFrameVisualizer = GameFrameVisualizer()
        self.human_mode = human_mode
        self.use_yolo = use_yolo
        
        self.yolo_model = None
        self.agent = None

        self.grid_window = None
        self.is_grid_shown = False

        if self.use_yolo:
            self.yolo_model = Yolo_Model(self.x_pixel_num, self.y_pixel_num)
            # self.yolo_model.set_logging()
        
        if self.human_mode:
            self.input_device = HumanInput()
        else:

            input_dim = (15 * 16 * 4 + 3) * 4 # (15 * 16 * 4 + 3(마리오 상태) ) * 4 : 3852
            hidden_dims = [1024, 256]
            output_dim = 12                   # action 12개
            self.agent = PPOAgent(input_dim, hidden_dims, output_dim)
            self.input_device = AgentInput(self.agent)


            self.game_thread = threading.Thread(target=self.game.run)
            self.game_thread.start()

    def init_grid_window(self, grid_window):
        self.is_grid_shown = True
        self.grid_window = grid_window


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
                    if current_time - last_update_time < update_interval:
                        continue
                    yolo_input_img = self.game.get_yolo_input_img()
                    mario_state = self.game.get_mario_state() # 0: small, 1: big,  => 2 : firey

                    tensor = self.yolo_model.get_tensor(yolo_input_img, mario_state)
                    # tensor = self.game.get_tensor()
                    # 누적 프레임만큼 쌓이지 않으면 tensor = None
                    # TODO 이렇게 되면 달리기를 절대 할 수 없는데 괜찮은가?
                    if tensor != None:
                        action = self.input_device.get_action(tensor)
                        self.game.receive_action(action)
                        last_update_time = current_time
                
                self.show_grid()

                self.clock.tick(self.fps) # 이 부분을 수정하면 fps 제한 없이 연산, 업데이트 를 반복하며 프레임 손실 없이 가능
        finally:
            if not self.human_mode:
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
    window = GridWindow()
    window.show()
    main = Main(human_mode=True, use_yolo = False)
    main.init_grid_window(window)

    main.run()

