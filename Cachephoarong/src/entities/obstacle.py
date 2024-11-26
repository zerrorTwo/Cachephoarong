import pygame
import random
import os
from ..constants import *

class Obstacle:
    def __init__(self):
        self.positions = []
        # Lấy đường dẫn tuyệt đối đến thư mục assets
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        image_path = os.path.join(current_dir, 'assets', 'obstacle4.png')
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (GRIDSIZE, GRIDSIZE))
        self.randomize_positions()

    def randomize_positions(self):
        self.positions = []
        for _ in range(20):  # số lượng vật cản
            while True:
                x = random.randint(0, GRID_WIDTH - 1) * GRIDSIZE
                y = random.randint(0, GRID_HEIGHT - 1) * GRIDSIZE

                # Kiểm tra chỗ đặt vật cản
                if (x, y) != (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2):
                    self.positions.append((x, y))
                    break

    def draw(self, surface):
        for pos in self.positions:
            # Vẽ hình ảnh thay vì hình chữ nhật
            surface.blit(self.image, (pos[0], pos[1]))