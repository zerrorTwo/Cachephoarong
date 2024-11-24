import pygame
import random
from ..constants import *


class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = SNAKE_BODY_COLOR

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self, grid, obstacles):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x * GRIDSIZE))), (cur[1] + (y * GRIDSIZE)))

        # Kiểm tra va chạm giữa con rắn và chướng ngại vật
        if (
            new in obstacles
            or new[0] < 0
            or new[1] < 0
            or new[0] >= SCREEN_WIDTH
            or new[1] >= SCREEN_HEIGHT
        ):
            return False

        # Kiểm tra va chạm giữa con rắn và chính nó
        if len(self.positions) > 2 and new in self.positions[2:]:
            return False

        self._update_grid(grid, new)

        return True

    def _update_grid(self, grid, new_pos):
        # cập nhật bằng cách xoá vị trí cũ đi
        if len(self.positions) > self.length:
            old = self.positions.pop()
            grid[int(old[1] / GRIDSIZE)][int(old[0] / GRIDSIZE)] = EMPTY

        # Thêm vị trí mới của con rắn
        self.positions.insert(0, new_pos)
        grid[int(new_pos[1] / GRIDSIZE)][int(new_pos[0] / GRIDSIZE)] = SNAKE_HEAD

        for pos in self.positions[1:]:
            grid[int(pos[1] / GRIDSIZE)][int(pos[0] / GRIDSIZE)] = SNAKE_BODY

        # Cập nhật đuôi
        self.tail = self.positions[-1]
        grid[int(self.tail[1] / GRIDSIZE)][int(self.tail[0] / GRIDSIZE)] = SNAKE_TAIL

    def draw(self, surface):
        for index, pos in enumerate(self.positions):
            r = pygame.Rect((pos[0], pos[1]), (GRIDSIZE, GRIDSIZE))
            if index == 0:  # đầu con rắn
                pygame.draw.rect(surface, SNAKE_HEAD_COLOR, r)
            elif index == self.length - 1:  # đuôi con rắn
                pygame.draw.rect(surface, (0, 230, 255), r)
            else:  # Vẽ thân con rắn
                color = (
                    abs(240 - 4 * index),
                    abs(240 - 4 * index),
                    abs(240 - 4 * index),
                )
                pygame.draw.rect(surface, color, r)
            pygame.draw.rect(surface, SNAKE_BORDER, r, 1)

    def reset(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
