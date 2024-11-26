from .node import Node
import sys
sys.setrecursionlimit(10000)  # Tăng giới hạn đệ quy

def backtracking(start_pos, goal_pos, grid, obstacles):
    start_node = Node(start_pos)
    goal_node = Node(goal_pos)
    visited = set()
    path = []
    
    def backtrack(current_node, goal_node):
        if current_node == goal_node:
            return True
            
        visited.add(current_node.position)
        
        for neighbor in current_node.get_neighbors(grid, obstacles):
            if neighbor.position not in visited:
                path.append(neighbor.position)
                if backtrack(neighbor, goal_node):
                    return True
                path.pop()
                
        return False

    path.append(start_node.position)
    if backtrack(start_node, goal_node):
        return path
    return None
