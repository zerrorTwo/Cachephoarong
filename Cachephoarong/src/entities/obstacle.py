import pygame
import random
from ..constants import *

class Obstacle:
    def __init__(self):
        self.positions = []
        self.color = OBSTACLE_COLOR
        self.randomize_positions()

    def randomize_positions(self):
        self.positions = []
        for _ in range(0):  # số lượng vật cản
            while True:
                x = random.randint(0, GRID_WIDTH - 1) * GRIDSIZE
                y = random.randint(0, GRID_HEIGHT - 1) * GRIDSIZE

                # Kiểm tra chỗ đặt vật cản
                if (x, y) != (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2):
                    self.positions.append((x, y))
                    break

    def draw(self, surface):
        for pos in self.positions:
            r = pygame.Rect((pos[0], pos[1]), (GRIDSIZE, GRIDSIZE))
            pygame.draw.rect(surface, self.color, r)