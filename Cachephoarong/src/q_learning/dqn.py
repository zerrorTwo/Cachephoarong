import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import glob
import re

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.episode_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_state(self, game):
        # Lấy vị trí đầu rắn
        head_x, head_y = game.snake.get_head_position()
        head_x = head_x / game.grid.shape[0]  # Chuẩn hóa
        head_y = head_y / game.grid.shape[1]

        # Lấy vị trí thức ăn
        food_x, food_y = game.food.position
        food_x = food_x / game.grid.shape[0]  # Chuẩn hóa
        food_y = food_y / game.grid.shape[1]

        # Tạo ma trận danger cho 8 hướng xung quanh đầu rắn
        danger = [0] * 8
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i, (dx, dy) in enumerate(directions):
            check_x = head_x + dx
            check_y = head_y + dy
            # Kiểm tra va chạm với tường hoặc thân rắn
            if (check_x < 0 or check_x >= game.grid.shape[0] or 
                check_y < 0 or check_y >= game.grid.shape[1] or
                (check_x, check_y) in game.snake.positions[:-1]):
                danger[i] = 1

        state = np.array([head_x, head_y, food_x, food_y] + danger)
        return torch.FloatTensor(state).to(self.device)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.stack([s[0] for s in minibatch])
        actions = torch.tensor([s[1] for s in minibatch]).to(self.device)
        rewards = torch.tensor([s[2] for s in minibatch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([s[3] for s in minibatch])
        dones = torch.tensor([s[4] for s in minibatch], dtype=torch.float32).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save_checkpoint(self, score, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory,
            'score': score,
            'episode': self.episode_count  # Thêm biến đếm episode
        }
        torch.save(checkpoint, path)
        print(f"Đã lưu checkpoint với điểm số: {score}")

    def load_checkpoint(self, path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = checkpoint['memory']
        self.episode_count = checkpoint.get('episode', 0)  # Lấy số episode đã train
        
        print(f"Đã tải checkpoint với epsilon: {self.epsilon}")
        return checkpoint['score']