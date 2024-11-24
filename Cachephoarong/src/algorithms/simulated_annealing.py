import random
import math
from .node import Node


# Hàm tính khoảng cách bằng Manhattan
def heuristic(node, goal_node):
    return abs(node.position[0] - goal_node.position[0]) + abs(
        node.position[1] - goal_node.position[1]
    )


def simulated_annealing(start_pos, goal_pos, grid, obstacles):
    init_temporature = 1000
    final_temp = 1  # nhiệt độ khi thuật toán dừng
    alpha = 0.95  # hệ số làm mát

    current = Node(start_pos)
    goal = Node(goal_pos)

    path_curr = [current]
    path_best = path_curr
    best_cost = float("inf")

    temp = init_temporature
    while temp > final_temp:
        curr_node = path_curr[-1]
        neighbors = curr_node.get_neighbors(grid, obstacles)

        if not neighbors:
            break

        next_node = random.choice(neighbors)
        new_path = path_curr + [next_node]

        current_cost = len(path_curr) + heuristic(path_curr[-1], goal)
        new_cost = len(new_path) + heuristic(next_node, goal)

        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
            path_curr = new_path

            if new_cost < best_cost:
                path_best = new_path
                best_cost = new_cost

        if next_node == goal:
            return [node.position for node in path_best]

        temp *= alpha

    if path_best:
        return [node.position for node in path_best]
    return None
