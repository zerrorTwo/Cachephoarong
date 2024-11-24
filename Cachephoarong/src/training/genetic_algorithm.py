import copy
import random
import numpy as np
import os
import pickle
from ..game import Game
from .neural_network import NeuralNetwork

class GeneticAlgorithm:
    def __init__(self, population_size=1000, generation_number=100):
        self.population_size = population_size
        self.generation_number = generation_number
        self.mutation_rate = 0.7
        self.elite_size = 30
        self.tournament_size = 8
        self.population = []
        self.best_score = 0
        self.best_network = None
        self.save_dir = "models"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Khởi tạo quần thể
        for _ in range(population_size):
            network = NeuralNetwork(
                input_size=24,
                hidden_size=16,
                output_size=4
            )
            self.population.append(network)

    def save_model(self, generation):
        if self.best_network is not None:
            model_path = os.path.join(self.save_dir, f"best_model_gen_{generation}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_network, f)
            
            info_path = os.path.join(self.save_dir, f"training_info_gen_{generation}.txt")
            with open(info_path, 'w') as f:
                f.write(f"Generation: {generation}\n")
                f.write(f"Best Score: {self.best_score}\n")
                f.write(f"Population Size: {self.population_size}\n")
                f.write(f"Network Architecture: [24, 16, 4]\n")

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.best_network = pickle.load(f)
        return self.best_network

    def train(self):
        no_improvement_count = 0
        best_score_threshold = self.best_score
        
        for generation in range(self.generation_number):
            print(f"Generation {generation + 1}")
            
            # Điều chỉnh mutation_rate theo thế hệ
            self.mutation_rate = max(0.3 - generation * 0.0005, 0.05)
            
            # Đánh giá từng cá thể
            fitness_scores = []
            for idx, network in enumerate(self.population):
                # Hiển thị thông tin về cá thể đang được đánh giá
                print(f"\rTesting individual {idx + 1}/{self.population_size}", end="")
                
                # Tạo game mới với chế độ hiển thị
                game = Game(ai_mode=True, display_game=True)  # Hiển thị giao diện khi train
                score = game.run(network)
                network.score = score
                fitness_scores.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_network = copy.deepcopy(network)
                    print(f"\nNew best score: {self.best_score}")
                    self.save_model(generation + 1)
            
            print("\n")  # Xuống dòng sau khi hoàn thành một thế hệ
            
            # Early stopping nếu không cải thiện
            if self.best_score > best_score_threshold:
                best_score_threshold = self.best_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= 30:  # Giảm từ 50 xuống 30 thế hệ
                print("Early stopping triggered")
                break
                
            # Tăng mutation_rate nếu score không cải thiện
            if no_improvement_count > 15:  # Giảm từ 20 xuống 15
                self.mutation_rate = min(self.mutation_rate * 1.2, 0.8)  # Điều chỉnh tăng nhẹ nhàng hơn
            
            # Tạo thế hệ mới
            new_population = []
            
            # Giữ lại cá thể tốt nhất (elitism)
            best_network = max(self.population, key=lambda x: x.score)
            new_population.append(copy.deepcopy(best_network))
            
            # Tạo phần còn lại của quần thể
            while len(new_population) < self.population_size:
                # Chọn cha mẹ
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                
                # Lai ghép
                child = self._crossover(parent1, parent2)
                
                # Đột biến
                child = self._mutate(child)
                
                new_population.append(child)
            
            self.population = new_population

    def _select_parent(self):
        tournament_size = self.tournament_size
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.score)

    def _crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
        
        for i in range(len(child.weights)):
            mask = np.random.rand(*child.weights[i].shape) < 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
            
        for i in range(len(child.biases)):
            mask = np.random.rand(*child.biases[i].shape) < 0.5
            child.biases[i] = np.where(mask, parent1.biases[i], parent2.biases[i])
            
        return child

    def _mutate(self, network):
        mutation_strength = 0.2  # Điều chỉnh độ mạnh của mutation
        
        # Tăng mutation_strength nếu score thấp
        if network.score < self.best_score * 0.5:
            mutation_strength *= 2
            
        for i in range(len(network.weights)):
            mask = np.random.rand(*network.weights[i].shape) < self.mutation_rate
            network.weights[i] += mask * np.random.randn(*network.weights[i].shape) * mutation_strength
            
        return network