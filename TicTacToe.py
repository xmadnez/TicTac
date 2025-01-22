class TicTacToe:
    def __init__(self):
        self.reset()

    def is_valid_move(self, position):
        return 0 <= position < 9 and self.board[position] == 0 and not self.game_over

    def make_move(self, position):
        if self.is_valid_move(position):
            self.board[position] = self.current_player
            self.check_winner()
            self.switch_player()
            return 0
        else:
            return -1

    def get_valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0:
                self.game_over = True
                return self.board[combo[0]]

        if 0 not in self.board:
            self.game_over = True
            return 0

        return None

    def switch_player(self):
        self.current_player = 1 if self.current_player == -1 else -1


    def reset(self):
        self.board = [0 for _ in range(9)]
        self.current_player = -1
        self.game_over = False
        return self.board
    