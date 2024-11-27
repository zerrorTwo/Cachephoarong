import numpy as np
import random
from ..constants import GRIDSIZE, GRID_WIDTH, GRID_HEIGHT
import glob
import re
import os

# ví dụ bảng gồm có 10 * 10 ô thì Q table sẽ có 10 * 10 * 10 * 10 = 10000 hàng và 4 cột
# vì đầu con rắn ở toạ độ x có 10 toạ độ y có 10 toạ độ,  có 10 toạ độ x của thức ăn, có 10 toạ độ y của thức ăn
# nên số lượng trạng thái có thể có là 10 * 10 * 10 * 10 = 10000

class QLearning:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.1 # tốc độ học của agent
        self.discount_factor = 0.95 # cân nhắc về số điểm của phần thưởng
        self.epsilon = 0.1 # tý lệ khám phá ngẫu nhiên
        
        # Tạo thư mục models nếu chưa tồn tại
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Tìm file q_table có điểm số cao nhất
        q_table_files = glob.glob(os.path.join(self.model_dir, 'q_table_*.npy'))
        best_score = -1
        best_q_table = None
        
        for file in q_table_files:
            score = int(re.findall(r'q_table_(\d+).npy', os.path.basename(file))[0])
            if score > best_score:
                best_score = score
                best_q_table = file
        
        if best_q_table:
            self.q_table = np.load(best_q_table)
            print(f"Đã tải Q-table từ file {os.path.basename(best_q_table)}")
        else:
            print("Tạo Q-table mới")
            self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])
        
    # cập nhật Q-table với số điểm thưởng
    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.amax(self.q_table[next_state])
        
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + self.learning_rate * target
        
        # giảm tỷ lệ khám phá ngẫu nhiên sau mỗi lần
        if self.epsilon > 0.01:
            self.epsilon *= 0.995
    def get_state(self, game):
        head_x, head_y = game.snake.get_head_position()
        food_x, food_y = game.food.position
        
        head_x = int(head_x / GRIDSIZE)
        head_y = int(head_y / GRIDSIZE)
        food_x = int(food_x / GRIDSIZE)
        food_y = int(food_y / GRIDSIZE)
        
        state = head_x + head_y * GRID_WIDTH + food_x * GRID_WIDTH * GRID_HEIGHT + food_y * GRID_WIDTH * GRID_HEIGHT * GRID_WIDTH
        return state