import pygame
import retro
import numpy as np
from utils import SMB

class Game():
    def __init__(self,x_pixel_num,y_pixel_num):
        # self.env = retro.make(game='SuperMarioBros-Nes', state = 'Level3-1')
        self.env = retro.make(game='SuperMarioBros-Nes', state = 'Level1-1')
        self.env.reset()
        
        # Pygame 설정
        pygame.init()
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.game_screen = pygame.display.set_mode((self.x_pixel_num, self.y_pixel_num))
        # self.game_screen = pygame.display.set_mode((512, 480))
        self.clock = pygame.time.Clock()


        self.ram = None

    def update_game(self, action):
        # action이 np.array([0] * 9) 형태가 아닌 경우 에러 발생
        if not isinstance(action, np.ndarray) or action.shape != (9,):
            raise ValueError("The action must be a numpy array of shape (9,).")

        # # Keys correspond with             B, NULL, SELECT, START, U, D, L, R, A
        # # index                            0  1     2       3      4  5  6  7  8
        # self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)



        # 게임 환경 업데이트
        obs, rew, done, info = self.env.step(action)
        if done:
            self.env.reset()

        # 게임 화면 업데이트
        # self.game_screen.blit(pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1)), (0, 0))

        # 게임 화면 스케일링
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        self.game_screen.blit(frame, (0, 0))
        pygame.display.flip()
        
        # 타일 정보 업데이트
        self.ram = self.env.get_ram()
    
    def get_ram(self):
        return self.ram
    
    def is_recordable(self):
        return SMB.is_recordable(self.ram)
    
    def get_mario_state(self):
        # 0: small, 1: big,  => 2 : firey
        return SMB.get_mario_state(self.ram)
    
    def get_frame(self) -> pygame.surface: 
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        return frame
