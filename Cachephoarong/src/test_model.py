from training.genetic_algorithm import GeneticAlgorithm
from game import Game

def test_trained_model(model_path):
    # Tải model đã train
    ga = GeneticAlgorithm()
    network = ga.load_model(model_path)
    
    # Chạy game với model đã train
    game = Game(ai_mode=True)
    score = game.run(network)
    print(f"Test Score: {score}")

if __name__ == "__main__":
    # Thay đổi đường dẫn tới model bạn muốn test
    model_path = "models/best_model_gen_100.pkl"
    test_trained_model(model_path) 