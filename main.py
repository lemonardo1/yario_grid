from Game import Game
from InputType import HumanInput, AgentInput
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from Visualizer import GameFrameVisualizer
import numpy as np
from PyQt5.QtWidgets import QApplication
import time
import pygame
import sys

class Main:
    def __init__(self):
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.game = Game(self.x_pixel_num, self.y_pixel_num)
        self.humanInput = HumanInput()
        self.agentInput = AgentInput()
        self.gameFrameVisualizer = GameFrameVisualizer()
        self.action = np.array([0] * 9)

        self.isHuman = True  # 이 값이 참이면 사람이 플레이하는 모드, 입력을 키보드로 받음
        self.running = True
        self.fps = 60
        self.frame_duration = 1 / self.fps

    def get_input(self):
        self.action = self.humanInput.get_action()
        print(self.action)

    def update_game(self):
        # print('Update game')
        self.get_input()
        self.game.update_game(self.action)
        frame = self.game.get_frame()
        self.gameFrameVisualizer.visualize(frame)

    def run(self):
        last_time = time.time()
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - last_time

            if elapsed_time >= self.frame_duration:
                self.update_game()
                last_time = current_time


class Main1:
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
        
        if self.human_mode:
            self.input_device = HumanInput()
        else:
            self.input_device = AgentInput()

    def run(self):
        running = True
        while running:
            if self.human_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                action = self.input_device.get_action()
                self.game.update_game(action)
                frame = self.game.get_frame()
                self.gameFrameVisualizer.visualize(frame)
                pygame.display.flip()
                self.clock.tick(self.fps)  # FPS 설정
            else:
                # 에이전트 모드일 때 게임 로직
                action = self.input_device.get_action()
                self.game.update_game(action)
                frame = self.game.get_frame()
                self.gameFrameVisualizer.visualize(frame)
                pygame.display.flip()
                self.clock.tick(self.fps) # 이 부분을 수정하면 fps 제한 없이 연산, 업데이트 를 반복하며 프레임 손실 없이 가능


if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication 객체 생성
    main = Main1(human_mode=False)
    main.run()