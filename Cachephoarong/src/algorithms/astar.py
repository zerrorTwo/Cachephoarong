import heapq 
from .node import Node
from ..constants import *

# Hàm tính khoảng cách bằng Manhattan
def heuristic(node, goal_node):
    return abs(node.position[0] - goal_node.position[0]) + \
        abs(node.position[1] - goal_node.position[1])

def a_star(start_pos, goal_pos, grid, obstacles):
    open_list = [] # danh sách nút có thể đi
    visited = [] # danh sách nút đã đi

    start_node = Node(start_pos)
    goal_node = Node(goal_pos)

    # Dùng heapq để có thể lấy các đường có cost nhỏ nhất
    heapq.heappush(open_list, (0 + heuristic(start_node, goal_node), start_node))

    g_costs = {start_node: 0}
    h_costs = {}
    traces = {}

    while open_list:
        _, current_node = heapq.heappop(open_list)
        visited.append(current_node)

        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = traces[current_node]
            path.append(start_node.position)
            return path[::-1]

        for child in current_node.get_neighbors(grid, obstacles):
            if child in visited:
                continue

            g_cost = g_costs[current_node] + 1
            h_cost = heuristic(child, goal_node)
            f_cost = g_cost + h_cost

            if child not in g_costs or g_cost < g_costs[child]:
                g_costs[child] = g_cost
                h_costs[child] = h_cost
                traces[child] = current_node
                heapq.heappush(open_list, (f_cost, child))

    return None