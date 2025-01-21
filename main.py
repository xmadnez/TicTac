import pygame
import sys
import random

# Farben und Dimensionen
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 15
CELL_SIZE = WIDTH // 3


# Spiellogik
class TicTacToe:
    def __init__(self):
        self.reset()

    def is_valid_move(self, position):
        return 0 <= position < 9 and self.board[position] == ' ' and not self.game_over

    def make_move(self, position):
        if self.is_valid_move(position):
            self.board[position] = self.current_player
            self.check_winner()
            self.switch_player()

    def get_valid_moves(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                self.game_over = True
                return self.board[combo[0]]

        if ' ' not in self.board:
            self.game_over = True
            return 'Draw'

        return None

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def ai_move(self, temperature):
        if not self.game_over:
            return self.make_move(random.choice(self.get_available_moves))

    def reset(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.game_over = False


# Zeichnet das Spielfeld und die Spielsteine
def draw_board(screen, game):
    screen.fill(WHITE)
    # Linien zeichnen
    for i in range(1, 3):
        pygame.draw.line(screen, BLACK, (0, CELL_SIZE * i), (WIDTH, CELL_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (CELL_SIZE * i, 0), (CELL_SIZE * i, HEIGHT), LINE_WIDTH)

    # Steine zeichnen
    for i, cell in enumerate(game.board):
        x = (i % 3) * CELL_SIZE
        y = (i // 3) * CELL_SIZE
        if cell == 'X':
            pygame.draw.line(screen, RED, (x + 30, y + 30), (x + CELL_SIZE - 30, y + CELL_SIZE - 30), LINE_WIDTH)
            pygame.draw.line(screen, RED, (x + CELL_SIZE - 30, y + 30), (x + 30, y + CELL_SIZE - 30), LINE_WIDTH)
        elif cell == 'O':
            pygame.draw.circle(screen, BLUE, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 2 - 30, LINE_WIDTH)


# Spielfeldposition bestimmen basierend auf Mausposition
def get_board_position(mouse_pos):
    x, y = mouse_pos
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row * 3 + col


# Hauptprogramm
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tic Tac Toe")
    clock = pygame.time.Clock()
    game = TicTacToe()

    # Spielmodi
    player_vs_ai = False  # True: Spieler vs. KI, False: KI vs. KI
    running = True
    temperature = 0 # ist ein wert zwischen 0 und 1. Je höher der Wert, 
        #desto wahrscheinlicher ist es, dass eine schlechte Option von der KI verwendet wird. 

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Spielerinteraktion bei Spieler vs. KI
            if player_vs_ai and game.current_player == 'X':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = get_board_position(pygame.mouse.get_pos())
                    game.make_move(pos)

        # KI-Zug
        if not player_vs_ai or game.current_player == 'O':
            pygame.time.delay(500)  # KI-Delay für realistischeres Verhalten
            move = game.ai_move(temperature)

        # Spielfeld zeichnen
        draw_board(screen, game)
        pygame.display.flip()

        # Spielende
        if game.game_over:
            print("Spielende!")
            print(f"Gewinner: {game.check_winner()}" if game.check_winner() != 'Draw' else "Unentschieden!")
            pygame.time.wait(2000)
            game.reset()
            if not player_vs_ai:
                running = False  # Spielende bei KI vs. KI

        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
