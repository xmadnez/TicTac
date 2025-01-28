import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from TicTacToe import TicTacToe

# Check if CUDA (GPU) is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TicTacToe Neural Network Model
class TicTacToeNN(nn.Module):

    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)  # Output a 9-dimensional vector for 9 possible moves
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000) 
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Raw Q-values, no activation here
        return x
    
    def train(self, memory, batch_size=32, gamma=0.9):
        self.memory += memory
        if len(self.memory) >= batch_size:
            batch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in batch:
                state_b_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                next_state_b_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Compute Q-values and targets
                q_values: torch.tensor = model(state_b_tensor)
                next_q_values: torch.tensor = model(next_state_b_tensor)
                
                target = reward + (gamma * torch.max(next_q_values).item() * (1 - done))
                current_q_value = q_values[0][action]
                
                # Compute loss and update weights
                loss = self.criterion(current_q_value, torch.tensor(target, dtype=torch.float32).to(device))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss.item()
            

model: TicTacToeNN = None

def init():
    PATH = "tictactoe_model.pth"
    global model
    if model is None:
        model = TicTacToeNN()
        model.to(device)  # Ensure the model is on the correct device (GPU or CPU)

        if os.path.exists(PATH):
            model.load_state_dict(torch.load(PATH))
            model.eval()
            model.to(device)
        else:
            train_model(model, episodes=1000, temperature=0.01)


# Training the model using Q-learning
def train_model(model: TicTacToeNN, episodes, epsilon=0.1, temperature=0.01):
    env = TicTacToe()

    for episode in range(episodes):
        memory_p1 = deque(maxlen=10000)
        memory_p2 = deque(maxlen=10000)    
        state = env.reset()
        while not env.game_over:
            
            # Epsilon-greedy action selection with temperature-controlled softmax
            if random.random() < epsilon:
                action = random.choice(env.get_valid_moves())
            else:
                action = model_turn(model, env.board, temperature)
            
            # Take the action and observe the result
            reward = env.make_move(action)
            if env.game_over:
                winner = env.check_winner()
                if env.current_player == -1:
                    mem = list(memory_p1.pop())
                    mem[2] = winner * env.current_player
                    mem[4] = True
                    memory_p1.append(tuple(mem))
                    reward += winner * -env.current_player
                else:
                    mem = list(memory_p2.pop())
                    mem[2] = winner * env.current_player
                    mem[4] = True
                    memory_p2.append(tuple(mem))
                    reward += winner * -env.current_player

            
            # Store the experience in the replay buffer
            (memory_p1 if env.current_player == -1 else memory_p2).append((state, action, reward, env.board, env.game_over))

            state = env.board
        model.train(memory_p1)
        loss = model.train(memory_p2)

        
            
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Loss: {loss}")
    
    # Save the trained model weights
    torch.save(model.state_dict(), 'tictactoe_model.pth')
    print("Model training completed and weights saved.")

def get_matrix_with_model(model: TicTacToeNN, state, temperature):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, 9)
    q_values: torch.tensor = model(state_tensor)
    
    if temperature == 0:
        # If temperature is 0, choose the action with the highest Q-value
        return q_values[0].detach().cpu().numpy()
    else:
        # Apply softmax with temperature (avoid NaN or infinity)
        q_values = q_values[0].detach().cpu().numpy()  # Move to CPU for further processing
        q_values = q_values / temperature
        exp_values = np.exp(q_values - np.max(q_values))  # Subtract max to prevent overflow
        exp_values = np.clip(exp_values, 0, np.inf)  # Clip any large values to prevent overflow
        
        # Normalize the probabilities to ensure they sum to 1
        prop_values = exp_values / np.sum(exp_values)  

        # Return the probabilities for all actions
        return prop_values

# Function to let the model make a decision (based on current model)
def model_turn(model: TicTacToeNN, state, temperature=0.01):
    val = get_matrix_with_model(model, state, temperature)
    if temperature == 0:
        # return np.argmax(val)
        valid_actions = [i for i in range(9) if state[i] == 0]
        return valid_actions[np.argmax(val[valid_actions])]
    else:
        # return np.random.choice(9, p=val)
        valid_actions = [i for i in range(9) if state[i] == 0]
        # all valid actions are 0 so just return first valid action
        if not np.any(val[valid_actions]):
            return valid_actions[0]
        prop = val[valid_actions] / np.sum(val[valid_actions])
        return np.random.choice(valid_actions, p=prop)
    
def get_matrix(state, temperature):
    global model
    if model == None:
        init()
    return  get_matrix_with_model(model, state, temperature)

def move(game: TicTacToe, temperature):
    global model
    if model == None:
        init()
    if not game.game_over:
        for i in range(100):
            move = model_turn(model, game.board, temperature)
            print(f"ai wants to move to {move}")
            if game.is_valid_move(move):
                return game.make_move(move)
        print(get_matrix(game.board, temperature))
        raise Exception("ai choose only invalid moves for 100 times")


