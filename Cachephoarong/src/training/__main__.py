from .genetic_algorithm import GeneticAlgorithm
import logging

def mainAI():
    """
    Khởi tạo và chạy thuật toán di truyền liên tục.
    population_size: 50
    generation_number: 100
    """
    logging.info("Bắt đầu huấn luyện thuật toán di truyền...")
    
    while True:  # Vòng lặp vô hạn để train liên tục
        try:
            ga = GeneticAlgorithm(population_size=1000, generation_number=100)
            ga.train()
            logging.info("Hoàn thành một chu kỳ huấn luyện, bắt đầu chu kỳ mới...")
        except KeyboardInterrupt:
            logging.info("Đã dừng huấn luyện theo yêu cầu người dùng")
            break
        except Exception as e:
            logging.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
            continue

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mainAI() 