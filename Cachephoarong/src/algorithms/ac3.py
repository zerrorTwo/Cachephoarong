from collections import deque
from .node import Node
from ..constants import *
import heapq

# Tìm các miền cảu 1 nút
def get_domain(pos, grid_size, obstacles, grid):
    domain = []
    x, y = pos
    for dx, dy in [UP, RIGHT, DOWN, LEFT]:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < grid_size[0] and
                0 <= new_y < grid_size[1] and
                (new_x * GRIDSIZE, new_y * GRIDSIZE) not in obstacles and
                grid[new_y][new_x] not in [SNAKE_BODY, SNAKE_HEAD, SNAKE_TAIL]):
            domain.append((dx, dy))
    return domain


def ac3(start_pos, goal_pos, grid_size, obstacles, grid):
    current = Node(start_pos)

    open_list = [(current, 0)]
    visited = {current}
    parent = {}

    while open_list:
        curr_node, curr_cost = open_list.pop(0)

        if curr_node.position == goal_pos:
            path = []
            while curr_node in parent:
                path.append(curr_node.position)
                curr_node = parent[curr_node]
            path.append(start_pos)
            return path[::-1]

        current_domain = get_domain(curr_node.position, grid_size, obstacles, grid)

        for dx, dy in current_domain:
            next_pos = (curr_node.position[0] + dx, curr_node.position[1] + dy)
            next_node = Node(next_pos)

            if next_node not in visited:
                visited.add(next_node)
                parent[next_node] = curr_node

                h_cost = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])

                i = 0
                while (i < len(open_list) and
                       h_cost > abs(open_list[i][0].position[0] - goal_pos[0]) +
                       abs(open_list[i][0].position[1] - goal_pos[1])):
                    i += 1
                open_list.insert(i, (next_node, curr_cost + 1))

    return None