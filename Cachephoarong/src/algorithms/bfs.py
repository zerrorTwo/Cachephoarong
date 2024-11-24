from collections import deque
from .node import Node

def bfs(start_pos, goal_pos, grid, obstacles):
    open_list = deque() # danh sách các nút đang xét
    visited = set() # danh sách các nút đã xét rồi

    start_node = Node(start_pos) # Nút khởi đầu là đầu con rắn
    goal_node = Node(goal_pos) # vị trí thức ăn

    open_list.append(start_node)
    visited.add(start_node)

    while open_list:
        current_node = open_list.popleft() # lấy nút đầu trong danh sách

        # Nếu nó tới được đồ ăn thì trả về đường đi
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.append(start_node.position)
            return path[::-1]

        # Lấy các nút mà con rắn có thể di chuyển
        for child in current_node.get_neighbors(grid, obstacles):
            if child not in visited:
                visited.add(child)
                open_list.append(child)

    return None