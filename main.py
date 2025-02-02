import pygame
import sys
from TicTacToe import TicTacToe

# Farben und Dimensionen
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
WIDTH, HEIGHT = 600, 700  # Zusätzlicher Platz für Slider und Buttons
LINE_WIDTH = 15
CELL_SIZE = WIDTH // 3

# Zeichnet das Spielfeld und die Spielsteine
def draw_board(screen: pygame.Surface, game, show_debug, temperature):
    screen.fill(WHITE)
    for i in range(1, 3):
        pygame.draw.line(screen, BLACK, (0, CELL_SIZE * i), (WIDTH, CELL_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (CELL_SIZE * i, 0), (CELL_SIZE * i, WIDTH), LINE_WIDTH)

    if show_debug:
        matrix = ai.get_matrix(game.board, temperature)
        
    for i, cell in enumerate(game.board):
        x = (i % 3) * CELL_SIZE
        y = (i // 3) * CELL_SIZE
        if show_debug:
            font = pygame.font.Font(None, 36)
            label = font.render(f"{matrix[i]:.6f}" if matrix[i]!=0 else 0, 1, BLACK)
            screen.blit(label, (x + 3 , y + CELL_SIZE // 2))
            
        if cell == -1:  # X
            pygame.draw.line(screen, RED, (x + 30, y + 30), (x + CELL_SIZE - 30, y + CELL_SIZE - 30), LINE_WIDTH)
            pygame.draw.line(screen, RED, (x + CELL_SIZE - 30, y + 30), (x + 30, y + CELL_SIZE - 30), LINE_WIDTH)
        elif cell == 1:  # O
            pygame.draw.circle(screen, BLUE, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 2 - 30, LINE_WIDTH)

# Zeichnet den Slider, die Buttons und die Gewinneranzeige
def draw_controls(screen, temperature, winner_message=None):
    font = pygame.font.Font(None, 36)

    # Temperatur-Slider
    temp_label = font.render(f"temp {temperature}", True, BLACK)
    
    screen.blit(temp_label, (20, WIDTH + 10))
    pygame.draw.rect(screen, GRAY, (200, WIDTH + 20, 200, 10))
    pygame.draw.circle(screen, BLACK, (200 + int(temperature * 200), WIDTH + 25), 10)

    give_up_rect = pygame.Rect(450, WIDTH + 20, 120, 50)
    pygame.draw.rect(screen, RED, give_up_rect)
    
    # give_up_label = font.render("Reset", True, WHITE)
    give_up_label = font.render("Aufgeben", True, WHITE)
    screen.blit(give_up_label, (448, WIDTH + 30))

    new_game_rect = None
    if winner_message:
        new_game_rect = pygame.Rect(200, WIDTH + 80, 200, 50)
        pygame.draw.rect(screen, BLUE, new_game_rect)
        new_game_label = font.render("Neues Spiel", True, WHITE)
        screen.blit(new_game_label, (WIDTH // 2 - new_game_label.get_width() // 2, WIDTH + 90))

    return give_up_rect, new_game_rect

# Gewinnernachricht in der Mitte des Spielfelds zeichnen
def draw_winner_message(screen, winner_message):
    font = pygame.font.Font(None, 72)
    winner_label = font.render(winner_message, True, BLACK, WHITE)
    screen.blit(winner_label, (WIDTH // 2 - winner_label.get_width() // 2, HEIGHT // 2 - winner_label.get_height() // 2))

# Startbildschirm für Spielmodi-Auswahl
def draw_start_screen(screen):
    screen.fill(WHITE)
    font = pygame.font.Font(None, 50)

    title = font.render("Tic Tac Toe", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

    player_vs_player_rect = pygame.Rect(WIDTH // 2 - 150, 250, 300, 50)
    pygame.draw.rect(screen, BLUE, player_vs_player_rect)
    player_vs_player_label = font.render("Gegen Spieler", True, WHITE)
    screen.blit(player_vs_player_label, (WIDTH // 2 - player_vs_player_label.get_width() // 2, 260))

    player_vs_ai_rect = pygame.Rect(WIDTH // 2 - 150, 350, 300, 50)
    pygame.draw.rect(screen, RED, player_vs_ai_rect)
    player_vs_ai_label = font.render("Gegen KI", True, WHITE)
    screen.blit(player_vs_ai_label, (WIDTH // 2 - player_vs_ai_label.get_width() // 2, 360))

    return player_vs_player_rect, player_vs_ai_rect

# Spielfeldposition bestimmen
def get_board_position(mouse_pos):
    x, y = mouse_pos
    if y >= WIDTH:  
        return None
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row * 3 + col

# Hauptprogramm
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tic Tac Toe")
    clock = pygame.time.Clock()
    
    running = True
    
    game_mode = None  # Auswahl zwischen "player_vs_player" und "player_vs_ai"
    temperature = 0.5  
    winner_message = None  
    show_debug = True

    # Startbildschirm für die Spielmodus-Wahl
    while running and game_mode is None:
        player_vs_player_rect, player_vs_ai_rect = draw_start_screen(screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if player_vs_player_rect.collidepoint(event.pos):
                    game_mode = "player_vs_player"
                elif player_vs_ai_rect.collidepoint(event.pos):
                    game_mode = "player_vs_ai"

    # Spiellogik starten
    game = TicTacToe()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Spielerinteraktion (Spieler vs Spieler)
            if winner_message is None and game_mode == "player_vs_player":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = get_board_position(event.pos)
                    if pos is not None:
                        game.make_move(pos)

            # Spielerinteraktion (Spieler vs KI)
            if winner_message is None and game_mode == "player_vs_ai" and game.current_player == -1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = get_board_position(event.pos)
                    if pos is not None:
                        game.make_move(pos)

            # Temperatur-Slider
            if event.type == pygame.MOUSEBUTTONDOWN and winner_message is None:
                x, y = event.pos
                if 200 <= x <= 400 and WIDTH + 20 <= y <= WIDTH + 40:
                    temperature = (x - 200) / 200

            # Aufgeben-Button
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if 450 <= x <= 570 and WIDTH + 20 <= y <= WIDTH + 70:
                    game.reset()
                    winner_message = None

            # "Neues Spiel"-Button
            if event.type == pygame.MOUSEBUTTONDOWN and winner_message is not None:
                x, y = event.pos
                if 200 <= x <= 400 and WIDTH + 80 <= y <= WIDTH + 130:
                    game.reset()
                    winner_message = None

        # KI-Zug (nur wenn Spieler vs KI)
        if winner_message is None and game_mode == "player_vs_ai" and game.current_player == 1:
            pygame.time.delay(500)
            ai.move(game, temperature)

        # Spielende prüfen
        if game.game_over and winner_message is None:
            winner = game.check_winner()
            if winner == -1:
                winner_message = "Spieler gewinnt!"
            elif winner == 1:
                winner_message = "KI gewinnt!"
            else:
                winner_message = "Unentschieden!"

        # Spielfeld zeichnen
        draw_board(screen, game, show_debug, temperature)
        
        if winner_message:
            draw_winner_message(screen, winner_message)
        draw_controls(screen, temperature, winner_message)
        pygame.display.flip()

        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()


