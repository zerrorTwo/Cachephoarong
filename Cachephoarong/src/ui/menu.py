import pygame
import sys
from ..constants import *


class Menu:
    @staticmethod
    def show_main_menu(screen):
        """Hiển thị menu chính và trả về thuật toán được chọn"""
        button_font = pygame.font.SysFont("monospace", 16, bold=True)

        algorithms = {
            'BFS': "Thuật Toán BFS",
            'A*': "Thuật Toán A*",
            'AC3': "Thuật Toán AC3",
            'SA': "Thuật Toán SA",
            'COMPARE': "So Sánh"
        }

        buttons = {}
        y_offset = SCREEN_HEIGHT // 2 - 120

        running = True
        while running:

            for algo, text in algorithms.items():
                button_text = button_font.render(text, True, WHITE)
                button = pygame.Rect(
                    (SCREEN_WIDTH // 2 - 100, y_offset),
                    (200, 50)
                )
                pygame.draw.rect(screen, ORANGE, button)
                screen.blit(
                    button_text,
                    button_text.get_rect(center=button.center)
                )
                buttons[algo] = button
                y_offset += 60

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for algo, button in buttons.items():
                        if button.collidepoint(event.pos):
                            return algo

            y_offset = SCREEN_HEIGHT // 2 - 120

    @staticmethod
    def show_pause_menu(screen):
        """Hiển thị menu tạm dừng"""
        font = pygame.font.SysFont("monospace", 32)

        text = font.render("Game Paused!", True, WHITE)
        screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 4))

        continue_text = font.render("Continue", True, WHITE)
        continue_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50, 200, 50)
        pygame.draw.rect(screen, ORANGE, continue_button)
        screen.blit(continue_text, continue_text.get_rect(center=continue_button.center))

        restart_text = font.render("Restart", True, WHITE)
        restart_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 50)
        pygame.draw.rect(screen, ORANGE, restart_button)
        screen.blit(restart_text, restart_text.get_rect(center=restart_button.center))

        pygame.display.update()

        return continue_button, restart_button