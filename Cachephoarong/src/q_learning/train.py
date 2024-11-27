# run in terminal python -m src.q_learning.train to train
from src.game import Game
from src.q_learning.qlearning import QLearning
from ..constants import *
import numpy as np
import pygame
import time
import os

def train():
   # Tạo đường dẫn đến thư mục models
   model_dir = os.path.join(os.path.dirname(__file__), 'models')
   os.makedirs(model_dir, exist_ok=True)
   
   # Tải điểm cao nhất từ file nếu tồn tại
   best_score_path = os.path.join(model_dir, 'best_score.npy')
   try:
       best_score_data = np.load(best_score_path)
       best_score = int(best_score_data)
   except:
       best_score = 0
   
   game = Game(display_game=False)
   state_size = GRID_WIDTH * GRID_HEIGHT * GRID_WIDTH * GRID_HEIGHT
   action_size = 4  # UP, DOWN, LEFT, RIGHT
   agent = QLearning(state_size, action_size)
   
   episodes = 1000000
   max_steps = 100000
   
   for episode in range(episodes):
       game.reset_game()
       state = agent.get_state(game)
       total_reward = 0
       
       for step in range(max_steps):
        #    game.draw()
        #    pygame.display.flip()
        #    time.sleep(0.05)
           
           action = agent.get_action(state)
           
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
               reward = -1
               done = False
           
           next_state = agent.get_state(game)
           agent.update(state, action, reward, next_state, done)
           state = next_state
           total_reward += reward
           
           if done:
               break
       if episode % 1 == 0:
           print(f"Episode: {episode}, Score: {game.score}, Total Reward: {total_reward}")
       
       # Chỉ lưu khi đạt điểm số cao hơn điểm cao nhất mọi thời đại
       if game.score > best_score:
           best_score = game.score
           # Lưu q_table với tên file chứa điểm số
           q_table_path = os.path.join(model_dir, f'q_table_{best_score}.npy')
           np.save(q_table_path, agent.q_table)
           np.save(best_score_path, np.array(best_score))
           print(f"Đã lưu Q-table mới với điểm số cao nhất: {best_score}")
   
   # Xóa dòng lưu q-table ở cuối hàm train
if __name__ == "__main__":
   train()