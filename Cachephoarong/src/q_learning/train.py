# run in terminal python -m src.q_learning.train to train
import glob
from src.game import Game
from src.q_learning.dqn import DQNAgent
from ..constants import *
import numpy as np
import pygame
import time
import os
import torch

def train():
   # Tạo đường dẫn đến thư mục models
   model_dir = os.path.join(os.path.dirname(__file__), 'models')
   os.makedirs(model_dir, exist_ok=True)
   
   game = Game(display_game=False)
   state_size = 12  # 4 cho vị trí (head_x, head_y, food_x, food_y) + 8 cho danger directions
   action_size = 4  # UP, DOWN, LEFT, RIGHT
   agent = DQNAgent(state_size, action_size)
   
   best_score = 0
   checkpoint_files = glob.glob(os.path.join(model_dir, 'dqn_checkpoint_*.pth'))
   if checkpoint_files:
       latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
       best_score = agent.load_checkpoint(latest_checkpoint)
   
   episodes = 100000
   max_steps = 100000
   
   for episode in range(episodes):
       agent.episode_count = episode
       game.reset_game()
       state = agent.get_state(game)
       total_reward = 0
       
       for step in range(max_steps):
        #    game.draw()
        #    pygame.display.flip()
        #    time.sleep(0.05)
           
           action = agent.act(state)
           
           # Chuyển đổi action thành hướng di chuyển
           direction = [UP, DOWN, LEFT, RIGHT][action]
           game.snake.turn(direction)
           
           if not game.snake.move(game.grid, game.obstacles.positions):
               reward = -10
               done = True
           elif game.snake.get_head_position() == game.food.position:
               reward = 10
               game.snake.length += 1
               game.score = game.snake.length - 1
               game.food.randomize_position(game.grid, game.snake.positions, game.obstacles.positions)
               done = False
           else:
               reward = -0.1  # Phạt nhẹ cho mỗi bước đi để khuyến khích tìm đường ngắn nhất
               done = False
           
           next_state = agent.get_state(game)
           agent.remember(state, action, reward, next_state, done)
           agent.replay()
           state = next_state
           total_reward += reward
           
           if done:
               break
       if episode % 1 == 0:
           print(f"Episode: {episode}, Score: {game.score}, Total Reward: {total_reward}")
       
       # Chỉ lưu khi đạt điểm số cao hơn điểm cao nhất mọi thời đại
       if game.score > best_score:
           best_score = game.score
           checkpoint_path = os.path.join(model_dir, f'dqn_checkpoint_{best_score}.pth')
           agent.save_checkpoint(best_score, checkpoint_path)
           print(f"Đã lưu model mới với điểm số cao nhất: {best_score}")
   
   # Xóa dòng lưu q-table ở cuối hàm train
if __name__ == "__main__":
   train()