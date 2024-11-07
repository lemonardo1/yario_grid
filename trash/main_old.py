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


class Main:
    def __init__(self, human_mode=True):
        pygame.init()
        self.fps = 60
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.screen = pygame.display.set_mode((self.x_pixel_num, self.y_pixel_num))
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
            self.yolo_model.set_logging()

    def run(self):
        running = True
        while running:
            if self.human_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                action = self.input_device.get_action()
                self.game.update_game(action)
                mario_state = self.game.get_mario_state()
                print(f"matrio state: {mario_state}")
                frame = self.game.get_frame()
                self.gameFrameVisualizer.visualize(frame)
                pygame.display.flip()
                self.clock.tick(self.fps)  # FPS 설정

            else:
                # 에이전트 모드일 때 게임 로직
                yolo_input_img = self.game.get_yolo_input_img()
                mario_state = self.game.get_mario_state() # 0: small, 1: big,  => 2 : firey

                tensor = self.yolo_model.get_tensor(yolo_input_img, mario_state)

                # 누적 프레임만큼 쌓이지 않으면 tensor = None
                # TODO 이렇게 되면 달리기를 절대 할 수 없는데 괜찮은가?
                if tensor != None:
                    action = self.input_device.get_action(tensor)
                else:
                    action = self.input_device.get_null_action()

                
                self.game.update_game(action)
                frame = self.game.get_frame()
                self.gameFrameVisualizer.visualize(frame)
                pygame.display.flip()
                self.clock.tick(self.fps) # 이 부분을 수정하면 fps 제한 없이 연산, 업데이트 를 반복하며 프레임 손실 없이 가능


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

