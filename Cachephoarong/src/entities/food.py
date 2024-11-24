import pygame
import random
from ..constants import *

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = FOOD_COLOR

    def randomize_position(self, grid, snake_positions, obstacles):
        if self.position != (0, 0):  # Chỉ xóa vị trí cũ nếu không phải vị trí khởi đầu
            grid[int(self.position[1] / GRIDSIZE)][int(self.position[0] / GRIDSIZE)] = EMPTY

        while True:
            x = random.randint(0, GRID_WIDTH - 1) * GRIDSIZE
            y = random.randint(0, GRID_HEIGHT - 1) * GRIDSIZE

            if ((x, y) not in snake_positions and
                    (x, y) not in obstacles):
                self.position = (x, y)
                grid[int(y / GRIDSIZE)][int(x / GRIDSIZE)] = FOOD
                break

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]),
                        (GRIDSIZE, GRIDSIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, SNAKE_BORDER, r, 1)

    def get_position(self):
        return self.position