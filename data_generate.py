import sys
import numpy as np
import pygame
import retro
import glob
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect

from typing import Tuple, List, Optional
# from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType, Item


import os
import json
from enum import Enum
from PIL import Image, ImageDraw
import shutil

# TODO 현재 상태(게임 상태, 죽은 상태) 파악 변수 -> 이 값에 따라 데이터 저장
# TODO 현재 게임 프레임 저장
# TODO 아이템(버섯들) 불러오는법


def draw_border(painter: QPainter, size: Tuple[float, float]) -> None:
    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
    painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
    painter.setRenderHint(QPainter.Antialiasing)
    points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    qpoints = [QPointF(point[0], point[1]) for point in points]
    polygon = QPolygonF(qpoints)
    painter.drawPolygon(polygon)


class Visualizer(QtWidgets.QWidget):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.size = size
        self.ram = None
        self.x_offset = 150
        self.tile_width, self.tile_height = 20, 20
        self.tiles = None
        self.enemies = None
        self._should_update = True

    def paintEvent(self, event):
        painter = QPainter(self)
        draw_border(painter, self.size)
        # if self.ram is not None:
        self.draw_tiles(painter)

        painter.end()

    def draw_tiles(self, painter: QPainter):
        
        if not self.tiles:
            return
        for row in range(15):
            for col in range(16):
                painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = 5 + (self.tile_width * col) + self.x_offset
                y_start = 5 + (self.tile_height * row)
                loc = (row, col)
                tile = self.tiles[loc]

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType, Item)):
                    # if tile.name == 'Empty' or tile.name == 'Fake':
                    #     continue
                    if ColorMap.has_name(tile.name):
                        rgb = ColorMap[tile.name].value
                        color = QColor(*rgb)
                    else:
                        # print(f"ColorMap has no color: {tile.name}")
                        rgb = (0,0,0)
                        color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    pass
                painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)

    def update_tiles(self, tiles):
        self.tiles = tiles
        self.update()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # self.env = retro.make(game='SuperMarioBros-Nes', state = 'Level3-1')
        self.env = retro.make(game='SuperMarioBros-Nes', state = 'Level1-1')
        self.env.reset()
        self.viz_window = Visualizer(self, (1100-514, 700))
        self.viz_window.setGeometry(0, 0, 1100-514, 700)
        self.setCentralWidget(self.viz_window)
        
        # Pygame 설정
        pygame.init()
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.game_screen = pygame.display.set_mode((self.x_pixel_num, self.y_pixel_num))
        # self.game_screen = pygame.display.set_mode((512, 480))
        self.clock = pygame.time.Clock()

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.start(1000 // 60)  # 60 FPS


        self.tiles = None

        self.data_dir = 'dataset'
        self.ensure_data_directory()
        self.frame_count = self.set_initial_frame_count()
        print(f"self.frame_count: {self.frame_count}")
    
    def ensure_data_directory(self):
        """Ensure the data directory exists and is ready for data storage."""
        paths = [
            self.data_dir,
            os.path.join(self.data_dir, 'images', 'all'),
            os.path.join(self.data_dir, 'images', 'all_boxed'),
            os.path.join(self.data_dir, 'labels', 'all')
        ]
        for path in paths:
            os.makedirs(path, exist_ok=True)

    def set_initial_frame_count(self):
        """Set the initial frame count based on existing files to avoid overwriting."""
        image_dir = os.path.join(self.data_dir, 'images', 'all')
        existing_files = glob.glob(os.path.join(image_dir, '*.png'))
        highest_count = 0
        for filename in existing_files:
            parts = os.path.basename(filename).split('_')
            if parts[0] == 'frame':
                frame_number = int(parts[1].replace('.png', ''))
                if frame_number > highest_count:
                    highest_count = frame_number
        return highest_count + 1  # Set to next number after highest
    
    def save_data(self, yolo_format):
        """Save the current game screen and tile data."""
        if self.env and self.tiles:
            formatted_frame_count = f"{self.frame_count:010}"
            image_base_path = os.path.join(self.data_dir, 'images', 'all', f"frame_{formatted_frame_count}.png")
            label_base_path = os.path.join(self.data_dir, 'labels', 'all', f"frame_{formatted_frame_count}.txt")
            image_boxed_base_path = os.path.join(self.data_dir, 'images', 'all_boxed', f"frame_{formatted_frame_count}.png")
            
            # Save the screen capture directly as PNG
            pygame.image.save(self.game_screen, image_base_path)
            
            # Save YOLO format data
            with open(label_base_path, 'w') as f:
                for label_value, coordinates in yolo_format.items():
                    for coordinate in coordinates:
                        x_yolo, y_yolo = coordinate[0]
                        x_unit_length, y_unit_length = coordinate[1]
                        f.write(f"{label_value} {x_yolo} {y_yolo} {x_unit_length} {y_unit_length}\n")


            with Image.open(image_base_path) as img:
                draw = ImageDraw.Draw(img)
                for label_value, coordinates in yolo_format.items():
                    for coordinate in coordinates:
                        x_yolo, y_yolo = coordinate[0]
                        x_unit_length, y_unit_length = coordinate[1]
                        img_width, img_height = img.size
                        x_pixel = int(x_yolo * img_width)
                        y_pixel = int(y_yolo * img_height)
                        box_width = x_unit_length * img_width
                        box_height = y_unit_length * img_height
                        box_coords = [
                            (x_pixel - box_width / 2, y_pixel - box_height / 2),
                            (x_pixel + box_width / 2, y_pixel + box_height / 2)
                        ]
                        draw.rectangle(box_coords, outline="red", width=1)
                img.save(image_boxed_base_path)

            self.frame_count += 1

    def update_game(self):
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        keys = pygame.key.get_pressed()
        action = np.array([0] * 9)  # 액션 초기화


        # # Keys correspond with             B, NULL, SELECT, START, U, D, L, R, A
        # # index                            0  1     2       3      4  5  6  7  8
        # self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        # 키 입력에 따라 액션 설정
        if keys[pygame.K_RIGHT]:
            action[7] = 1
        if keys[pygame.K_LEFT]:
            action[6] = 1
        if keys[pygame.K_SPACE]:
            action[8] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[0] = 1
        if keys[pygame.K_DOWN]:
            action[5] = 1
        if keys[pygame.K_UP]:
            action[4] = 1

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
        ram = self.env.get_ram()
        self.tiles = SMB.get_tiles(ram, detailed_enemies = True)
        self.viz_window.update_tiles(self.tiles)


        # SMB.groundTest(ram)
        # SMB.itemBoxTest(ram)
        yolo_format = SMB.get_yolo_format_new(ram)
        # yolo_format = SMB.get_yolo_format_unit_test(ram)
    
        # SMB.get_mario_state(ram)
        # SMB.get_Coins(ram)
        # SMB.get_World(ram)
        # SMB.get_Level(ram)
        # SMB.get_Lives(ram)
        # SMB.get_mario_score(ram)
        # SMB.get_Time(ram)
        SMB.get_item_pos(ram)

        if keys[pygame.K_0]:
            x_start = SMB.get_x_start(ram)
            print(x_start)
        
        # SMB.get_item_type(ram)
        
        
        if SMB.is_recordable(ram):
            self.save_data(yolo_format)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()