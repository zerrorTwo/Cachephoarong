import pygame
import sys
import numpy as np
from .constants import *
from .entities.snake import Snake
from .entities.food import Food
from .entities.obstacle import Obstacle
from .ui.menu import Menu
from .ui.grid import draw_grid
from .algorithms import astar, bfs, ac3, simulated_annealing
import matplotlib.pyplot as plt
from collections import defaultdict


class Game:
    def __init__(self, ai_mode=False, display_game=False):
        pygame.init()
        pygame.display.set_caption("Cá chép hoá rồng!!")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.ai_mode = ai_mode
        self.display_game = display_game

        self.reset_game()

    # Hàm dùng để reset game
    def reset_game(self):
        self.grid = init_grid()
        self.snake = Snake()
        self.obstacles = Obstacle()
        self.food = Food()
        self.food.randomize_position(
            self.grid, self.snake.positions, self.obstacles.positions
        )
        self.score = 0

    # Hàm để khởi động game, game có 2 chế độ là thuật toán và trai, ai là chế độ train
    def run(self, net=None):
        if self.ai_mode and self.display_game:
            running = True
            while running:
                self.clock.tick(FPS)
                if not self.handle_events():
                    break
                if not self.update(algorithm="AI", network=net):
                    break
                self.draw()
            return self.score

        while True:
            algorithm = Menu.show_main_menu(self.screen)
            if algorithm == "COMPARE":
                print("So sánh hiệu suất...")
                self.compare_algorithms()
                continue
            self.reset_game()
            running = True
            while running:
                self.clock.tick(FPS)
                if not self.handle_events():
                    break
                if not self.update(algorithm):
                    break
                self.draw()

    # Hàm dùng để vẽ có object
    def draw(self):
        self.surface.fill(BLACK)
        draw_grid(self.surface)
        self.obstacles.draw(self.surface)
        self.snake.draw(self.surface)
        self.food.draw(self.surface)
        self.screen.blit(self.surface, (0, 0))
        score_text = self.font.render(f"Score {self.score}", True, (255, 255, 0))
        self.screen.blit(score_text, (5, 10))
        pygame.display.update()

    # Hàm xử lí out game và pause game
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return self.pause_game()
        return True

    # Hàm dùng để dừng game tạm thời
    def pause_game(self):
        btn_continue, btn_restart = Menu.show_pause_menu(self.screen)
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return True

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_continue.collidepoint(event.pos):
                        return True
                    elif btn_restart.collidepoint(event.pos):
                        return False
        return True

    def update(self, algorithm=None, network=None):
        if self.ai_mode and network:
            state = self.get_state()
            output = network.forward(state)
            direction = self.get_direct(output)

            # Di chuyển theo hướng thuật toán đã chọn
            self.snake.turn(direction)
            if not self.snake.move(self.grid, self.obstacles.positions):
                return False

        else:
            # Lấy vị trí hiện tại là cái đầu con rắn
            start_pos = (
                self.snake.get_head_position()[0] / GRIDSIZE,
                self.snake.get_head_position()[1] / GRIDSIZE,
            )
            # Lấy vị trí đích là thức ăn
            goal_pos = (
                self.food.position[0] / GRIDSIZE,
                self.food.position[1] / GRIDSIZE,
            )

            # Tìm đường đi theo thuật toán được chọn
            path = None
            if algorithm == "A*":
                path = astar.a_star(
                    start_pos, goal_pos, self.grid, self.obstacles.positions
                )
            elif algorithm == "BFS":
                path = bfs.bfs(start_pos, goal_pos, self.grid, self.obstacles.positions)
            elif algorithm == "AC3":
                path = ac3.ac3(
                    start_pos,
                    goal_pos,
                    (GRID_WIDTH, GRID_HEIGHT),
                    self.obstacles.positions,
                    self.grid,
                )
            elif algorithm == "SA":
                path = simulated_annealing.simulated_annealing(
                    start_pos, goal_pos, self.grid, self.obstacles.positions
                )

            if path is None or len(path) < 2:
                return False

            # Tính hướng di chuyển từ đường đi
            next_pos = path[1]
            direction = (next_pos[0] - start_pos[0], next_pos[1] - start_pos[1])

            # Di chuyển rắn theo hướng đã tính toán
            self.snake.turn(direction)
            if not self.snake.move(self.grid, self.obstacles.positions):
                return False

        # Hàm dùng để kiểm tra đó có phải là thức ăn không, nếu phải thì độ dài con rắn tăng 1 và score tăng 1
        if self.snake.get_head_position() == self.food.position:
            self.snake.length += 1
            self.score = self.snake.length - 1
            self.food.randomize_position(
                self.grid, self.snake.positions, self.obstacles.positions
            )
        return True

    def get_state(self):
        state = np.zeros(24)
        return state.reshape(1, -1)

    def get_direct(self, output):
        direction_idx = np.argmax(output)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        return directions[direction_idx]

    def run_ai_mode(self, network):
        self.reset_game()
        running = True
        while running:
            self.clock.tick(FPS_AI)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if not self.update(algorithm="AI", network=network):
                break
            self.draw()

        return self.score

    # Hàm để so sánh các chỉ số
    def compare_algorithms(self):
        # Các thuật toán dùng để so sánh
        algorithms = ["BFS", "A*", "AC3", "SA"]
        # Số lần để so sánh
        num_of_algo = 10
        # tạo một defaultlist để lưu các chỉ số cần thiết

        stats = defaultdict(lambda: defaultdict(list))
        #      stats = {
        #     'AC3': {
        #         'scores': [],  # danh sách điểm số
        #         'moves': [],   # danh sách số bước di chuyển
        #         'time': []     # danh sách thời gian thực thi
        #     },
        # }
        for i in algorithms:
            for _ in range(num_of_algo):
                self.reset_game()
                score, moves, time_taken = self.run_algorithm(i)
                stats[i]["scores"].append(score)
                stats[i]["moves"].append(moves)
                stats[i]["time"].append(time_taken)

        # Tạo một đồ thị với 3 bảng
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Hiệu suất các thuật toán")

        # Vẽ
        metrics = {
            "scores": ("Điểm số", ax1),
            "moves": ("Số bước di chuyển", ax2),
            "time": ("Thời gian thực thi (giây)", ax3),
        }

        for metric, (ylabel, ax) in metrics.items():
            # Vẽ các đường cho từng thuật toán
            for algo in algorithms:
                values = stats[algo][metric]
                runs = range(1, len(values) + 1)
                ax.plot(runs, values, marker="o", label=algo)

            ax.set_xlabel("Lần chạy")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    # Hàm dùng để chạy thuật toán và đánh giá
    def run_algorithm(self, algo):
        import time
        start_time = time.time()
        moves = 0
        self.reset_game()
        running = True
        while running:  
            if not self.update(algorithm=algo):
                break
            moves += 1
            # nếu tham số display game là true thì sẽ hiện UI
            if self.display_game:
                self.draw()
                self.clock.tick(200)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
        time_taken = time.time() - start_time
        return self.score, moves, time_taken
