#C:/Users/NAM/AppData/Local/Programs/Python/Python313/python.exe D:\SPKT\AI\FolderMainSnake\ai-snake-project-2\test.py 
import pygame  # type: ignore
import sys
import random
import heapq
from collections import deque
import numpy as np # type: ignore # type: ignore
from pygame.display import update # type: ignore
from env import *
import math

class Obstacle(object):
    def __init__(self):
        self.positions = []
        self.color = (255, 0, 0)  # Màu đỏ cho chướng ngại vật
        self.randomize_positions()

    def randomize_positions(self):
        # Tạo một số lượng chướng ngại vật ngẫu nhiên
        self.positions = []
        for _ in range(10):  # Tạo 5 chướng ngại vật
            x = random.randint(0, GRID_WIDTH - 1) * GRIDSIZE
            y = random.randint(0, GRID_HEIGHT - 1) * GRIDSIZE
            self.positions.append((x, y))

    def draw(self, surface):
        for pos in self.positions:
            r = pygame.Rect((pos[0], pos[1]), (GRIDSIZE, GRIDSIZE))
            pygame.draw.rect(surface, self.color, r)

    def check_collision(self, position):
        return position in self.positions

class Snake(object):
    def __init__(self):
        self.length = 1 # chiều dài ban đầu bằng 0
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))] # xuất hiện giữa bảng
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT]) # các hướng con rắn có thể di chuyển
        self.color = (240, 240, 240)
        self.tail = (0, 0)

    #Hàm dùng để láy 
    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x*GRIDSIZE))), (cur[1] + (y*GRIDSIZE)))

        if cur in obs.positions or new[0] < 0 or new[1] < 0 or new[0] >= SCREEN_WIDTH or new[1] >= SCREEN_HEIGHT:
            dead()

        if (len(self.positions) > 2 and new in self.positions[2:-1]) or new[0] == -GRIDSIZE or new[1] == -GRIDSIZE or new[0] == SCREEN_WIDTH or new[1] == SCREEN_HEIGHT:
            self.reset()
            reset_grid()
            food.randomize_position()
        else:
            
            for i in self.positions:
                grid[int(i[1] / GRIDSIZE), int(i[0] / GRIDSIZE)] = 1
            
            grid[int(new[1] / GRIDSIZE), int(new[0] / GRIDSIZE)] = 3
            
            if len(self.positions) + 1 > self.length:
                old = self.positions.pop()
                grid[int(old[1] / GRIDSIZE), int(old[0] / GRIDSIZE)] = 0
            self.positions.insert(0, new)
            grid[int(self.positions[-1][1] / GRIDSIZE), int(self.positions[-1][0] / GRIDSIZE)] = 4
            self.tail = self.positions[-1]

    def reset(self):
        global score
        self.length = 1
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        score = 0
    
    def draw(self, surface):
        for index, p in enumerate(self.positions):
            r = pygame.Rect((p[0], p[1]), (GRIDSIZE, GRIDSIZE))
            if index == 0:
                pygame.draw.rect(surface, (230, 0, 255), r)
                pygame.draw.rect(surface, (93, 216, 228), r, 1)
                continue
            if index == snake.length - 1:
                pygame.draw.rect(surface, (0, 230, 255), r)
                pygame.draw.rect(surface, (93, 216, 228), r, 1)
                continue
            pygame.draw.rect(surface, (abs(240 - 4*index), abs(240 - 4*index), abs(240 - 4*index)), r)
            pygame.draw.rect(surface, (93, 216, 228), r, 1)

    def handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.turn(UP)
                elif event.key == pygame.K_DOWN:
                    self.turn(DOWN)
                elif event.key == pygame.K_LEFT:
                    self.turn(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.turn(RIGHT)
class Food(object):
    def __init__(self):
        self.position = (0, 0)
        self.color = (114, 137, 218)
        self.randomize_position()

    def get_position(self):
        return self.position

    def randomize_position(self):
        grid[int(self.position[1]/GRIDSIZE), int(self.position[0]/GRIDSIZE)] = 0
        self.position = (random.randint(0, GRID_WIDTH-1) * GRIDSIZE, random.randint(0, GRID_HEIGHT-1) * GRIDSIZE)
        if self.position in snake.positions:
            self.randomize_position() #recursive call
        if self.position in obs.positions:
            self.randomize_position() #recursive call
        grid[int(self.position[1]/GRIDSIZE), int(self.position[0]/GRIDSIZE)] = 2

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (GRIDSIZE, GRIDSIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, (93, 216, 228), r, 1)

    def get_position(self):
        return self.position

class Node():
    def __init__(self, position, parent = None):
        self.position = (int(position[0]), int(position[1]))
        self.g_cost = float('inf')  
        self.h_cost = float('inf')  
        self.f_cost = float('inf') 
        self.parent = parent
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        # Compare nodes based on f_cost
        return self.f_cost < other.f_cost

    def __repr__(self):
        return str(self.position)

    def get_parent(self):
        return self.parent

    def get_neighbors(self):
        parent_pos = self.position
        x = parent_pos[0]
        y = parent_pos[1]
        children = []
        for new_position in [UP, RIGHT, DOWN, LEFT]:
            node_position = (self.position[0] + new_position[0], self.position[1] + new_position[1])
            x, y = node_position
            if (x* GRIDSIZE, y* GRIDSIZE) in obs.positions:
                continue
            temp = self
            broken = False
            while temp is not None:
                if (temp.position == node_position):
                    broken = True
                    break
                temp = temp.parent
            if broken:
                continue
            if node_position[0] >= GRID_WIDTH or node_position[0] < 0 or node_position[1] >= GRID_HEIGHT or node_position[1] < 0:
                continue
            if (grid[node_position[1], node_position[0]] == 1 or grid[node_position[1], node_position[0]] == 3 or (grid[node_position[1], node_position[0]] == 4)):
                continue
            new_node = Node(node_position, self)
            children.append(new_node)
        return children

    def on_grid(self):
        x = self.position[0]
        y = self.position[1]
        return (x >= GRID_WIDTH or x < 0 or y >= GRID_HEIGHT or y < 0)
    
    def is_snake_node(self, snake):
        x = self.position[0]
        y = self.position[1]
        for pos in snake.positions:
            if (pos[0] == x and pos[1] == y):
                return True
        return False
    
    def is_obstacle_node(self, obs):
        return self.position in obs.positions

    
    def get_position(self):
        return self.position
    
    def listify(self):
        list = []
        list.insert(0, self.position)
        node = self.parent
        while node is not None:
            list.insert(0, node.position)
            node = node.parent
        return list
    def __hash__(self):
        return hash(self.position)

def drawGrid(surface, myfont):
    for y in range(0, int(GRID_HEIGHT)):
        for x in range(0, int(GRID_WIDTH)):
            if (x + y) % 2 == 0:
                r = pygame.Rect((x*GRIDSIZE, y*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
                pygame.draw.rect(surface, (44, 47, 51), r)
            else:
                rr = pygame.Rect((x*GRIDSIZE, y*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
                pygame.draw.rect(surface, (35, 39, 42), rr)

def heuristic(node, goal_node):
        return abs(node.get_position()[0] - goal_node.get_position()[0]) + \
               abs(node.get_position()[1] - goal_node.get_position()[1])

def a_star(start_pos, goal_pos):
    open_list = []  # Danh sách các nút cần xét
    visited = []  # Danh sách các nút đã xét
    

    start_node = Node(start_pos)
    goal_node = Node(goal_pos)
    
    # Hàm heuristic, tính khoảng cách Manhattan giữa hai điểm
    

    # Dùng heapq để làm danh sách open_list, mỗi phần tử là một tuple (f(n), node)
    heapq.heappush(open_list, (0 + heuristic(start_node, goal_node), start_node))  # f(n) = g(n) + h(n)

    # Mảng lưu trữ g_cost, h_cost, và trace
    g_costs = {start_node: 0}  # Chi phí đi từ điểm bắt đầu
    h_costs = {}
    traces = {}

    while open_list:
        # Lấy nút có f(n) nhỏ nhất từ open_list
        _, current_node = heapq.heappop(open_list)
        visited.append(current_node)

        if current_node == goal_node:
            # Truy vết đường đi từ goal về start
            path = []
            while current_node != start_node:
                path.append(current_node.get_position())
                current_node = traces[current_node]  # Sử dụng trace để truy vết đường đi
            path.append(start_node.get_position())
            return path[::-1]  # Đảo ngược đường đi để từ start đến goal

        # Lấy các láng giềng của nút hiện tại
        current_node_neighbors = current_node.get_neighbors()

        for child in current_node_neighbors:
            if child in visited:
                continue  # Bỏ qua nếu child đã được xét

            # Tính chi phí g(n) (cost so far)
            g_cost = g_costs[current_node] + 1  # Chi phí giữa các nút là 1 (có thể thay đổi)
            h_cost = heuristic(child, goal_node)  # Tính chi phí heuristic
            f_cost = g_cost + h_cost  # f(n) = g(n) + h(n)

            if child not in g_costs or g_cost < g_costs[child]:  # Nếu child chưa có trong open_list hoặc có chi phí g_cost thấp hơn
                g_costs[child] = g_cost
                h_costs[child] = h_cost
                traces[child] = current_node  # Lưu trace
                heapq.heappush(open_list, (f_cost, child))
            elif g_cost == g_costs[child]:  # Nếu chi phí g_cost của child bằng chi phí hiện tại trong open_list
                if h_cost < h_costs[child]:  # Nếu chi phí heuristic của child nhỏ hơn
                    h_costs[child] = h_cost
                    traces[child] = current_node  # Cập nhật trace

    return None  # Nếu không tìm được đường đi

def dfs(start_pos, goal_pos):
    open_list = []
    closed_list = []
    start_node = Node(start_pos)
    goal_node = Node(goal_pos)
    open_list.append(start_node)
    while (len(open_list) != 0):
        cur_node = open_list.pop(-1)
        closed_list.append(cur_node)
        if (cur_node == goal_node):
            path = []
            while cur_node != start_node:
                path.append(cur_node.get_position())
                cur_node = cur_node.get_parent()
            path.append(start_node.get_position())

            return path[::-1]

        cur_node_neighbors = cur_node.get_neighbors()
        

        for child in cur_node_neighbors:
            x, y = child.get_position()
            if (((x*GRIDSIZE, y* GRIDSIZE)) in obs.positions):
                continue
            if child in closed_list:
                continue  

            if child not in open_list:
                open_list.insert(0, child)
    
    return None

def bfs(start_pos, goal_pos):
    #fake queue đẻ chạy thuật toán
    open_list = []
    #mảng lưu các vị trí đã đi qua
    visited = []

    start_node = Node(start_pos)
    goal_node = Node(goal_pos)

    open_list.append(start_node)

    while len(open_list) != 0:
        current_node = open_list.pop(0)  # Lấy phần tử đu tiên trong danh sách (thay vì phần tử cuối trong DFS)

        visited.append(current_node)
        
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.get_position())
                current_node = current_node.get_parent()
            path.append(start_node.get_position())

            return path[::-1]

        cur_x = (start_node.get_position())[0]
        cur_y = (start_node.get_position())[1]
        goal_x = goal_node.get_position()[0]
        goal_y = goal_node.get_position()[1]

        current_node_neighbors = current_node.get_neighbors()

        for child in current_node_neighbors:
            # Nếu nút là vật cản hoặc đã có trong visited thì bỏ qua
            if child in visited:
                continue

            # Nếu nút chưa có trong open_list, thêm vào open_list
            if child not in open_list:
                open_list.append(child)

    return None

# Thuật toán AC3 để tìm đường đi
def get_domain(pos, grid_size, obstacles):
    """Lấy miền giá trị hợp lệ cho một vị trí"""
    domain = []
    x, y = pos
    for dx, dy in [UP, RIGHT, DOWN, LEFT]:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < grid_size[0] and 
            0 <= new_y < grid_size[1] and 
            (new_x * GRIDSIZE, new_y * GRIDSIZE) not in obstacles and
            grid[int(new_y), int(new_x)] != 1 and 
            grid[int(new_y), int(new_x)] != 3):
            domain.append((dx, dy))
    return domain

def ac3(start_pos, goal_pos, grid_size, obstacles):
    """AC3 algorithm để tìm đường đi cho con rắn"""
    # Khởi tạo các biến
    queue = deque()
    domains = {}
    path = []
    
    # Tạo các node và miền giá trị ban ầu
    current = Node(start_pos)
    goal = Node(goal_pos)
    
    # Khởi tạo miền giá trị cho vị trí hiện tại và mục tiêu
    domains[current] = get_domain(start_pos, grid_size, obstacles)
    domains[goal] = get_domain(goal_pos, grid_size, obstacles)
    
    # Thêm các cung vào queue với cách tiếp cận tốt hơn
    open_list = [(current, 0)]  # (node, cost)
    visited = {current}
    parent = {}
    
    while open_list:
        current_node, current_cost = open_list.pop(0)
        
        if current_node.position == goal_pos:
            # Tạo đường đi
            path = []
            while current_node in parent:
                path.append(current_node.position)
                current_node = parent[current_node]
            path.append(start_pos)
            return path[::-1]
        
        # Lấy các hướng đi hợp lệ
        current_domain = get_domain(current_node.position, grid_size, obstacles)
        
        for dx, dy in current_domain:
            next_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            next_node = Node(next_pos)
            
            if next_node not in visited:
                visited.add(next_node)
                parent[next_node] = current_node
                
                # Tính khoảng cách Manhattan đến mục tiêu
                h_cost = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
                
                # Thêm vào open_list với priority dựa trên khoảng cách
                insert_idx = 0
                while (insert_idx < len(open_list) and 
                       h_cost > abs(open_list[insert_idx][0].position[0] - goal_pos[0]) + 
                       abs(open_list[insert_idx][0].position[1] - goal_pos[1])):
                    insert_idx += 1
                open_list.insert(insert_idx, (next_node, current_cost + 1))
    
    return None
def reset_grid():
    global grid
    grid = np.zeros((GRID_WIDTH, GRID_HEIGHT))
    grid = grid.astype(int)

reset_grid()

directions = []

def snake_directions(path):
    if path is None or len(path) < 2:  # Thêm kiểm tra độ dài path
        return []
    directions = []

    for i in range(len(path) - 1):
        direction_vector = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        directions.insert(0, direction_vector)
    return directions

def dead():
    # Reset the game state
    snake.reset()
    reset_grid()
    food.randomize_position()
    return menu()

def menu():
    white = (255, 255, 255)
    orange = (254, 110, 26)

    # Font chữ
    button_font = pygame.font.SysFont("monospace", 16, bold=True)
    running = True
    while running:
        # Tạo nút cho thuật toán DFS
        button_text_dfs = button_font.render("Thuật Toán DFS", True, white)
        button_dfs = pygame.Rect((SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 80), (200, 50))
        pygame.draw.rect(screen, orange, button_dfs)
        screen.blit(button_text_dfs, button_text_dfs.get_rect(center=button_dfs.center))

        # Tạo nút cho thuật toán BFS
        button_text_bsf = button_font.render("Thuật Toán BFS", True, white)
        button_bsf = pygame.Rect((SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 -20), (200, 50))
        pygame.draw.rect(screen, orange, button_bsf)
        screen.blit(button_text_bsf, button_text_bsf.get_rect(center=button_bsf.center))

        # Tạo nút cho thuật toán A*
        button_text_astar = button_font.render("Thuật Toán A*", True, white)
        button_astar = pygame.Rect((SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 40), (200, 50))
        pygame.draw.rect(screen, orange, button_astar)
        screen.blit(button_text_astar, button_text_astar.get_rect(center=button_astar.center))

        # Tạo nút cho thuật toán AC3
        button_text_ac3 = button_font.render("Thuật Toán ac3", True, white)
        button_ac3 = pygame.Rect((SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 100), (200, 50))
        pygame.draw.rect(screen, orange, button_ac3)
        screen.blit(button_text_ac3, button_text_ac3.get_rect(center=button_ac3.center))

        # Thêm nút cho thuật toán Simulated Annealing
        button_text_sa = button_font.render("Thuật Toán SA", True, white)
        button_sa = pygame.Rect((SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 160), (200, 50))
        pygame.draw.rect(screen, orange, button_sa)
        screen.blit(button_text_sa, button_text_sa.get_rect(center=button_sa.center))

        pygame.display.flip()  # Cập nhật màn hình

        # Lắng nghe sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_dfs.collidepoint(event.pos):
                    running = False
                    thuattoan = 'DFS' # Chọn thuật toán DFS
                    return thuattoan  # Quay lại với kết quả
                elif button_bsf.collidepoint(event.pos):
                    running = False
                    thuattoan = 'BFS'  # Chọn thuật toán BFS
                    return thuattoan  # Quay lại với kết quả
                elif button_astar.collidepoint(event.pos):
                    running = False
                    thuattoan = 'A*'  # Chọn thuật toán A*
                    return thuattoan  # Quay lại với kết quả
                elif button_ac3.collidepoint(event.pos):
                    running = False
                    thuattoan = 'AC3'  # Chọn thuật toán AC3
                    return thuattoan  # Quay lại với kết quả
                elif button_sa.collidepoint(event.pos):
                    running = False
                    thuattoan = 'SA'  # Chọn thuật toán Simulated Annealing
                    return thuattoan

def pause_game():
    # Tạm dừng game và hiển thị các lựa chọn "Continue" và "Restart"
    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                paused = False  # Thoát khỏi chế độ tạm dừng và tiếp tục game

            # Kiểm tra nếu nhấn vào nút "Continue"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if continue_button.collidepoint(event.pos):
                    paused = False  # Tiếp tục game khi chọn "Continue"
                elif restart_button.collidepoint(event.pos):
                    paused = False  # Quay lại menu khi chọn "Restart"
                    screen.fill((255, 246, 233))  # Làm sạch màn hình trớc khi quay lại menu
                    pygame.display.update()  # Cập nhật màn hình
                    return 'restart'  # Quay lại menu

        # Hiển thị menu tạm dừng
        screen.fill((0, 0, 0))  # Làm mờ màn hình khi tạm dừng
        font = pygame.font.SysFont("monospace", 32)
        text = font.render("Game Paused!", True, (255, 255, 255))
        screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 4))

        # Tạo nút "Continue"
        continue_text = font.render("Continue", True, (255, 255, 255))
        continue_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, 200, 50)
        pygame.draw.rect(screen, (254, 110, 26), continue_button)
        screen.blit(continue_text, continue_text.get_rect(center=continue_button.center))

        # Tạo nút "Restart"
        restart_text = font.render("Restart", True, (255, 255, 255))
        restart_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 50)
        pygame.draw.rect(screen, (254, 110, 26), restart_button)
        screen.blit(restart_text, restart_text.get_rect(center=restart_button.center))

        pygame.display.update()

def simulated_annealing(start_pos, goal_pos):
    """Thuật toán Simulated Annealing để tìm đường đi cho con rắn"""
    # Các tham số cho thuật toán
    initial_temp = 1000
    final_temp = 1
    alpha = 0.95  # Tốc độ làm mát
    
    # Tạo node bắt đầu và kết thúc
    current = Node(start_pos)
    goal = Node(goal_pos)
    
    # Khởi tạo đường đi hiện tại
    current_path = [current]
    best_path = current_path
    best_cost = float('inf')
    
    temp = initial_temp
    while temp > final_temp:
        # Tạo một đường đi mới bằng cách thay đổi ng
        current_node = current_path[-1]
        neighbors = current_node.get_neighbors()
        
        if not neighbors:
            break
            
        # Chọn một node láng giềng ngẫu nhiên
        next_node = random.choice(neighbors)
        new_path = current_path + [next_node]
        
        # Tính chi phí đường đi
        current_cost = len(current_path) + heuristic(current_path[-1], goal)
        new_cost = len(new_path) + heuristic(next_node, goal)
        
        # Tính xác suất chấp nhận đường đi mới
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
            current_path = new_path
            
            # Cập nhật đường đi tốt nhất
            if new_cost < best_cost:
                best_path = new_path
                best_cost = new_cost
        
        # Kiểm tra nếu đã đến đích
        if next_node == goal:
            return [node.position for node in best_path]
            
        # Giảm nhiệt độ
        temp *= alpha
    
    # Trả về đường đi tốt nhất tìm được
    if best_path:
        return [node.position for node in best_path]
    return None
def main():
    global score, food, snake, surface, screen, obs, clock
    pygame.init()
    pygame.display.set_caption("Cá chép hoá rồng!!")
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    myfont = pygame.font.SysFont("monospace", 16)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()

    # Hiển thị menu và chọn thuật toán
    thuattoan = menu()
    
    while True:
        # Khởi tạo trạng thái game
        obs = Obstacle()
        snake = Snake()
        food = Food()
        score = 0

        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    result = pause_game()  # Dừng game và hiển thị menu
                    if result == 'restart':
                        thuattoan = menu()  # Quay lại menu chính để chọn lại thuật toán
                        break  # Thoát khỏi vòng lặp game và quay lại menu chính

            # Kiểm tra nếu rắn ăn được thức ăn
            if snake.get_head_position() == food.get_position():
                snake.length += 1
                score = snake.length - 1
                food.randomize_position()

            # Vẽ các phần tử trong game
            obs.draw(surface)
            snake.draw(surface)
            food.draw(surface)
            screen.blit(surface, (0, 0))
            text = myfont.render(f"Score {score}", True, (255, 255, 0))
            screen.blit(text, (5, 10))
            drawGrid(surface, myfont)

            # Logic pathfinding
            start_pos = (snake.get_head_position()[0] / GRIDSIZE, snake.get_head_position()[1] / GRIDSIZE)
            food_pos = (food.get_position()[0] / GRIDSIZE, food.get_position()[1] / GRIDSIZE)

            # Dùng thuật toán được chọn
            if thuattoan == 'A*':
                path = a_star(start_pos, food_pos)
            elif thuattoan == 'DFS':
                path = dfs(start_pos, food_pos)
            elif thuattoan == 'BFS':
                path = bfs(start_pos, food_pos)
            elif thuattoan == 'AC3':
                path = ac3(start_pos, food_pos, (GRID_WIDTH, GRID_HEIGHT), obs.positions)
            elif thuattoan == 'SA':
                path = simulated_annealing(start_pos, food_pos)

            if path is None or len(path) < 2:  # Thêm kiểm tra path
                thuattoan = dead()
                break

            directions = snake_directions(path)
            if not directions:  # Kiểm tra danh sách directions có trống không
                thuattoan = dead()
                break
            
            snake_dir = directions.pop()
            snake.turn(snake_dir)
            snake.move()

            pygame.display.update()

if __name__ == "__main__":
    main()