from src.game import Game
from src.q_learning.qlearning import QLearning
from ..constants import *
import numpy as np
def train():
   game = Game(display_game=False)
   state_size = GRID_WIDTH * GRID_HEIGHT * GRID_WIDTH * GRID_HEIGHT
   action_size = 4  # UP, DOWN, LEFT, RIGHT
   agent = QLearning(state_size, action_size)
   
   episodes = 100000
   max_steps = 1000
   
   for episode in range(episodes):
       game.reset_game()
       state = agent.get_state(game)
       total_reward = 0
       
       for step in range(max_steps):
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
       if episode % 100 == 0:
           print(f"Episode: {episode}, Score: {game.score}, Total Reward: {total_reward}")
           agent.save_q_table()
        #    print(f"Đã lưu Q-table sau episode {episode}")
   
   # Lưu mô hình Q-learning
   np.save('q_table.npy', agent.q_table)
if __name__ == "__main__":
   train()