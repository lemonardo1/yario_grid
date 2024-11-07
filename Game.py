import pygame
import retro
import numpy as np
from utils import SMB
from PIL import Image
from Visualizer import GameFrameVisualizer

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

        self.is_new_action_received = False
        self.new_action = None
        self.ram = None
        self.fps = 60
        self.null_action = np.array( [0, 1,    0,      0,     0, 0, 0, 0, 0], np.int8)

        self.gameFrameVisualizer = GameFrameVisualizer()
        self.running = True
    
    def stop(self):
        self.running = False

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

        # # 게임 화면 스케일링
        # frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        # frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        # self.game_screen.blit(frame, (0, 0))
        # pygame.display.flip()

    def run(self):
        # current_time = pygame.time.get_ticks()
        self.running = True 
        while self.running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # print("game class quit")
                    self.running = False
            
            if self.is_new_action_received:
                self.update_game(self.new_action)
                self.is_new_action_received = False
            else:
                self.update_game(self.null_action)
            
            
            # last_update_time = current_time

            self.gameFrameVisualizer.visualize(self.get_frame())
            self.clock.tick(self.fps)

    def receive_action(self, action):
        # print("new action!!!")
        self.is_new_action_received = True
        self.new_action = action
        


    def is_recordable(self):
        ram = self.env.get_ram()
        return SMB.is_recordable(ram)
    
    def get_mario_state(self):
        # 0: small, 1: big,  => 2 : firey
        ram = self.env.get_ram()
        return SMB.get_mario_state(ram)

    # visualize할떄 필요한 frame 반환
    def get_frame(self) -> pygame.surface: 
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        return frame
    
    def get_yolo_input_img(self) -> Image:
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        frame = pygame.transform.scale(frame, (self.x_pixel_num, self.y_pixel_num))  # 스케일링
        pil_image = Image.frombytes('RGB', (self.x_pixel_num, self.y_pixel_num), pygame.image.tostring(frame, 'RGB'))
        return pil_image
    

