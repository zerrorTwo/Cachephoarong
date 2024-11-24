import pygame
from ..constants import *

def draw_grid(surface):
    for y in range(0, int(GRID_HEIGHT)):
        for x in range(0, int(GRID_WIDTH)):
            if (x + y) % 2 == 0:
                r = pygame.Rect(
                    (x*GRIDSIZE, y*GRIDSIZE),
                    (GRIDSIZE, GRIDSIZE)
                )
                pygame.draw.rect(surface, GRID_COLOR1, r)
            else:
                r = pygame.Rect(
                    (x*GRIDSIZE, y*GRIDSIZE),
                    (GRIDSIZE, GRIDSIZE)
                )
                pygame.draw.rect(surface, GRID_COLOR2, r)