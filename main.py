from Game import Game
from InputType import HumanInput, AgentInput
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from Visualizer import GameFrameVisualizer
from Yolo_Model import Yolo_Model
from PIL import Image
from network import PPOAgent

import numpy as np
from PyQt5.QtWidgets import QApplication
import time
import pygame
import sys
import threading


# TODO yolo train 데이터 저장하는 클래스도 구현해두기
# TODO 에이전트에게 전달하는 텐서 값을 yolo output이 아니라 게임 ram애서 받아온 정보로 하는 방식도 구현
class Main:
    def __init__(self, human_mode=True):
        pygame.init()
        self.fps = 60
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.clock = pygame.time.Clock()
        self.game = Game(self.x_pixel_num, self.y_pixel_num)
        self.gameFrameVisualizer = GameFrameVisualizer()
        self.human_mode = human_mode
        
        self.yolo_model = None
        self.agent = None
        
        if self.human_mode:
            self.input_device = HumanInput()
        else:

            input_dim = (15 * 16 * 4 + 3) * 4 # (15 * 16 * 4 + 3(마리오 상태) ) * 4 : 3852
            hidden_dims = [1024, 256]
            output_dim = 12                   # action 12개
            self.agent = PPOAgent(input_dim, hidden_dims, output_dim)
            self.input_device = AgentInput(self.agent)
            self.yolo_model = Yolo_Model(self.x_pixel_num, self.y_pixel_num)
            # self.yolo_model.set_logging()

        self.game_thread = threading.Thread(target=self.game.run)
        self.game_thread.start()

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
                    self.game.update_game(action)

                else:
                    # 에이전트 모드일 때 게임 로직
                    if current_time - last_update_time < update_interval:
                        continue
                    yolo_input_img = self.game.get_yolo_input_img()
                    mario_state = self.game.get_mario_state() # 0: small, 1: big,  => 2 : firey

                    tensor = self.yolo_model.get_tensor(yolo_input_img, mario_state)

                    # 누적 프레임만큼 쌓이지 않으면 tensor = None
                    # TODO 이렇게 되면 달리기를 절대 할 수 없는데 괜찮은가?
                    if tensor != None:
                        action = self.input_device.get_action(tensor)
                        self.game.receive_action(action)
                        last_update_time = current_time
                    # else:
                    #     action = self.input_device.get_null_action()
                    #     self.game.update_game(action)

                    

                self.clock.tick(self.fps) # 이 부분을 수정하면 fps 제한 없이 연산, 업데이트 를 반복하며 프레임 손실 없이 가능
        finally:
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
    main = Main(human_mode=False)
    main.run()

