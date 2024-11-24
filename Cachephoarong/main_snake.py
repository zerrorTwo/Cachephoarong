# python -m src.training
# python -m src.test_model

from src.game import Game

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()