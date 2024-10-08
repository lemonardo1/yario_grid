import sys
import numpy as np
import pygame
import retro
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect

from typing import Tuple, List, Optional
from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType



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

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                    rgb = ColorMap[tile.name].value
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
        self.env = retro.make(game='SuperMarioBros-Nes')
        self.env.reset()
        self.viz_window = Visualizer(self, (1100-514, 700))
        self.viz_window.setGeometry(0, 0, 1100-514, 700)
        self.setCentralWidget(self.viz_window)
        
        # Pygame 설정
        pygame.init()
        # self.game_screen = pygame.display.set_mode((256, 240))
        self.game_screen = pygame.display.set_mode((512, 480))
        self.clock = pygame.time.Clock()

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.start(1000 // 60)  # 60 FPS

    def update_game(self):
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        keys = pygame.key.get_pressed()
        action = np.array([0] * 9)  # 액션 초기화

        # 키 입력에 따라 액션 설정
        if keys[pygame.K_RIGHT]:
            action[7] = 1
        if keys[pygame.K_LEFT]:
            action[6] = 1
        if keys[pygame.K_SPACE]:
            action[8] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[0] = 1

        # 게임 환경 업데이트
        obs, rew, done, info = self.env.step(action)
        if done:
            self.env.reset()

        # 게임 화면 업데이트
        # self.game_screen.blit(pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1)), (0, 0))

        # 게임 화면 스케일링
        frame = pygame.surfarray.make_surface(self.env.render(mode='rgb_array').swapaxes(0, 1))
        frame = pygame.transform.scale(frame, (512, 480))  # 스케일링
        self.game_screen.blit(frame, (0, 0))
        pygame.display.flip()
        
        # 타일 정보 업데이트
        ram = self.env.get_ram()
        tiles = SMB.get_tiles(ram)
        self.viz_window.update_tiles(tiles)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()