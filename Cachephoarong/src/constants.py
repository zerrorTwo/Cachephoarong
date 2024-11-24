import numpy as np

# Kích thước màn hình
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 720
WINDOW_SIZE = SCREEN_WIDTH

# Kích thước grid
GRIDSIZE = 40
GRID_WIDTH = int(SCREEN_HEIGHT / GRIDSIZE)
GRID_HEIGHT = int(SCREEN_WIDTH / GRIDSIZE)

# Hướng di chuyển
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Màu sắc
WHITE = (255, 255, 255)
ORANGE = (254, 110, 26)
BLACK = (0, 0, 0)
GRID_COLOR1 = (44, 47, 51)
GRID_COLOR2 = (35, 39, 42)
SNAKE_HEAD_COLOR = (230, 0, 255)
SNAKE_BODY_COLOR = (240, 240, 240)
SNAKE_BORDER = (93, 216, 228)
FOOD_COLOR = (114, 137, 218)
OBSTACLE_COLOR = (255, 0, 0)

# Grid states
EMPTY = 0
SNAKE_BODY = 1
FOOD = 2
SNAKE_HEAD = 3
SNAKE_TAIL = 4

# Training constants
SHOW_TRAINING = True  # Set False nếu không muốn hiển thị quá trình training
POPULATION_SIZE = 1000  # Số lượng cá thể trong quần thể
MUTATION_RATE = 0.7   # Tỷ lệ đột biến
NUM_GENERATIONS = 100 # Số thế hệ tối đa

FPS = 60
FPS_AI = 200

def init_grid():
    """Khởi tạo grid trống"""
    grid = np.zeros((GRID_WIDTH, GRID_HEIGHT))
    return grid.astype(int)