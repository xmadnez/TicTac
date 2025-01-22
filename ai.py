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

model = None

def init():
    PATH = "tictactoe_model.pth"
    global model
    if model == None:
        model = TicTacToeNN()

        if os.path.exists(PATH):
            model.load_state_dict(torch.load(PATH))
            model.eval()
            model.to(device)  # Ensure the model is on the correct device (GPU or CPU)
        else:
            train_model(model, episodes=10000, temperature=1.0)

# TicTacToe Neural Network Model
class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)  # Output a 9-dimensional vector for 9 possible moves
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Raw Q-values, no activation here
        return x

# Training the model using Q-learning
def train_model(model: TicTacToeNN, episodes, epsilon=0.1, gamma=0.9, batch_size=32, temperature=1.0):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = deque(maxlen=10000)  # Replay buffer
    env = TicTacToe()

    for episode in range(episodes):
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
                reward += env.check_winner()
            
            # Store the experience in the replay buffer
            memory.append((state, action, reward, env.board, env.game_over))

            # Sample a batch from the replay buffer
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                for state_b, action_b, reward_b, next_state_b, done_b in batch:
                    state_b_tensor = torch.tensor(state_b, dtype=torch.float32).unsqueeze(0).to(device)
                    next_state_b_tensor = torch.tensor(next_state_b, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Compute Q-values and targets
                    q_values_b = model(state_b_tensor)
                    next_q_values_b = model(next_state_b_tensor)
                    
                    target = reward_b + (gamma * torch.max(next_q_values_b).item() * (1 - done_b))
                    current_q_value = q_values_b[0][action_b]
                    
                    # Compute loss and update weights
                    loss = criterion(current_q_value, torch.tensor(target, dtype=torch.float32).to(device))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            state = env.board
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Loss: {loss.item()}")
    
    # Save the trained model weights
    torch.save(model.state_dict(), 'tictactoe_model.pth')
    print("Model training completed and weights saved.")


# Load the trained model weights
def load_trained_model(model, path='tictactoe_model.pth'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        model.to(device)  # Ensure the model is on the correct device (GPU or CPU)
        print("Trained model loaded successfully.")
    else:
        print("No trained model found, starting from scratch.")

# Function to let the model make a decision (based on current model)
def model_turn(model, state, temperature=1.0):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, 9)
    q_values = model(state_tensor)
    
    if temperature == 0:
        # If temperature is 0, choose the action with the highest Q-value
        valid_actions = [i for i in range(9) if state[i] == 0]  # Find empty spots
        action = q_values[np.argmax(q_values)]
    else:
        # Mask the invalid actions (those that are not empty)
        masked_q_values = q_values[0].detach().cpu().numpy()  # Move to CPU for further processing
        # masked_q_values = np.array([masked_q_values[i] if i in valid_actions else -float('inf') for i in range(9)])
        # Apply softmax with temperature (avoid NaN or infinity)
        masked_q_values = masked_q_values / temperature
        exp_values = np.exp(masked_q_values - np.max(masked_q_values))  # Subtract max to prevent overflow
        exp_values = np.clip(exp_values, 0, np.inf)  # Clip any large values to prevent overflow
        
        
        
        # Normalize the valid probabilities to ensure they sum to 1
        prop_values = exp_values / np.sum(exp_values)  

        # Sample an action based on the probabilities corresponding to the valid actions
        action = np.random.choice(np.arange(9), p=prop_values)  # Choose based on the softmax probabilities
    
    return action

def move(game: TicTacToe, temperature):
    global model
    if model == None:
        init()
    if not game.game_over:
        for i in range(100):
            move = model_turn(model, game.board, temperature)
            if move in game.get_valid_moves():
                return game.make_move(move)
        raise Exception("ai choose only invalid moves for 100 times")
        

