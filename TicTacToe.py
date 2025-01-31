from Memory import Memory
import csv

class TicTacToe:
    def __init__(self, keepMemory=True):
        self.keepMemory = keepMemory
        self.reset()

    def is_valid_move(self, position):
        return 0 <= position < 9 and self.board[position] == 0 and not self.game_over

    def make_move(self, position):
        if self.is_valid_move(position):
            prevBoard = self.board.copy()
            self.board[position] = self.current_player
            self.saveMemory(prevBoard, position)
            self.check_winner()
            self.switch_player()
        else:
            if self.keepMemory:
                self.invalid_moves.append(Memory(self.board.copy(), position, -1, self.board.copy(), True))

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

        # self.memory_p1 = deque(maxlen=10000)
        # self.memory_p2 = deque(maxlen=10000)
        self.memory_p1 = list()
        self.memory_p2 = list()
        self.invalid_moves = list()
        return self.board
    
    def trainAi(self):
        import ai
        self.writeMemoryToFile()
        # batch_size=len(self.memory_p1)+len(self.memory_p2)
        ret = ai.getModel().train_with_list(self.memory_p1 + self.memory_p2 + self.invalid_moves)
        ai.getModel().save()
        return ret

    
    def writeMemoryToFile(self, filename='memory.csv'):
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for memory in self.memory_p1:
                    writer.writerow(memory.to_tupel())

                for memory in self.memory_p2:
                    writer.writerow(memory.to_tupel())
                    
                for memory in self.invalid_moves:
                    writer.writerow(memory.to_tupel())

    def saveMemory(self, prev_board, action, reward=0):
        winner = self.check_winner()
        if self.keepMemory:
            
            if self.game_over:
                if self.current_player == 1:
                    self.memory_p1[-1].done = True
                    self.memory_p1[-1].reward = winner * -self.current_player
                    # mem = list(self.memory_p1.pop())
                    # mem[2] = winner * -self.current_player
                    # mem[4] = True
                    # self.memory_p1.append(tuple(mem))
                    reward += winner * self.current_player
                else:
                    self.memory_p2[-1].done = True
                    self.memory_p2[-1].reward = winner * -self.current_player
                    # mem = list(self.memory_p2.pop())
                    # mem[2] = winner * -self.current_player
                    # mem[4] = True
                    # self.memory_p2.append(tuple(mem))
                    reward += winner * self.current_player
                
            if self.current_player == -1:
                # self.memory_p1.append((prev_board, action, reward, self.board, self.game_over))
                self.memory_p1.append(Memory(prev_board, action, reward, self.board.copy(), self.game_over))
            else: 
                # self.memory_p2.append((prev_board, action, reward, self.board,self.game_over))
                self.memory_p2.append(Memory(prev_board, action, reward, self.board.copy(), self.game_over))
