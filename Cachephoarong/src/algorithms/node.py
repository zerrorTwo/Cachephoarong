from ..constants import *

class Node:
    def __init__(self, position, parent=None):
        self.position = (int(position[0]), int(position[1]))
        self.g_cost = float("inf")
        self.h_cost = float("inf")
        self.f_cost = float("inf")
        self.parent = parent

    # so sánh vị trí của 2 nút khác nhau
    def __eq__(self, other):
        return self.position == other.position

    # so sánh chi phí của thuật toán 
    def __lt__(self, other):
        return self.f_cost < other.f_cost

    # định danh để làm key cho dictionary
    def __hash__(self):
        return hash(self.position)

    # Đi tìm các nút mà con rắn có thể đi
    def get_neighbors(self, grid, obstacles):
        neighbors = []
        for direction in [UP, RIGHT, DOWN, LEFT]:
            new_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
            if self._is_valid_position(new_pos, grid, obstacles):
                neighbors.append(Node(new_pos, self))
        return neighbors

    # Kiểm tra vị trí đi có hợp lệ không
    def _is_valid_position(self, pos, grid, obstacles):
        x, y = pos

        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return False

        if (x * GRIDSIZE, y * GRIDSIZE) in obstacles:
            return False

        if grid[y][x] in [SNAKE_BODY, SNAKE_HEAD, SNAKE_TAIL]:
            return False

        return True
