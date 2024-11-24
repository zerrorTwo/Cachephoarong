# Đây là file constant cần thiết 
# Chiều dài và chiều rộng của bảng 
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 720

# kích thước của 1 ô 
GRIDSIZE = 40
GRID_WIDTH = int(SCREEN_HEIGHT / GRIDSIZE) # 720/20 -> 36*36 ô
GRID_HEIGHT = int(SCREEN_WIDTH / GRIDSIZE)

# Các hướng có thể di chuyển
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

score = 0
grid = 0

food = 0
snake = 0
surface = 0
depth = 0 # Khởi tạo độ sâu tìm kiếm
MAX_DEPTH = 0  # Giới hạn chiều sâu tìm kiếm tối đa