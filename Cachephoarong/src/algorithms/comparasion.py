import time
import matplotlib.pyplot as plt
from .astar import a_star
from .bfs import bfs 
from .ac3 import ac3
from .simulated_annealing import simulated_annealing

def compare_algorithms(start_pos, goal_pos, grid, obstacles, grid_size, num_runs=10):
    algorithms = {
        'A*': a_star,
        'BFS': bfs,
        'AC3': ac3,
        'SA': simulated_annealing
    }
    
    results = {
        'execution_time': {},
        'path_length': {}
    }

    for name_algorithm, algo_func in algorithms.items():
        times = []
        lengths = []
        
        for _ in range(num_runs):
            start_time = time.time()
            path = algo_func(start_pos, goal_pos, grid, obstacles)
            end_time = time.time()
            
            execution_time = end_time - start_time
            path_length = len(path) if path else float('inf')
            
            times.append(execution_time)
            lengths.append(path_length)
        
        results['execution_time'][name_algorithm] = sum(times) / len(times)
        results['path_length'][name_algorithm] = sum(lengths) / len(lengths)
    
    return results

def plot_comparison(results):
    # Tạo subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot thời gian thực thi
    algorithms = list(results['execution_time'].keys())
    times = list(results['execution_time'].values())
    ax1.bar(algorithms, times)
    ax1.set_title('Thời gian thực thi trung bình')
    ax1.set_ylabel('Thời gian (giây)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot độ dài đường đi
    lengths = list(results['path_length'].values())
    ax2.bar(algorithms, lengths)
    ax2.set_title('Độ dài đường đi trung bình')
    ax2.set_ylabel('Số bước')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Sử dụng:
def run_comparison(start_pos, goal_pos, grid, obstacles, grid_size):
    results = compare_algorithms(start_pos, goal_pos, grid, obstacles, grid_size)
    plot_comparison(results)