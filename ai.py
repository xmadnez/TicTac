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
    
    def train_with_list(self, memories, batch_size=32, gamma=0.9):
        for memory in memories:
            self.memory.append(memory.to_tupel())
        return self.train_without_memory(batch_size=batch_size, gamma=gamma)

    def train_without_memory(self, batch_size=32, gamma=0.9):
        if len(self.memory) >= batch_size:
            batch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in batch:
                state_b_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                next_state_b_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Compute Q-values and targets
                q_values: torch.tensor = self(state_b_tensor)
                next_q_values: torch.tensor = self(next_state_b_tensor)
                
                target = reward + (gamma * torch.max(next_q_values).item() * (1 - done))
                current_q_value = q_values[0][action]
                
                # Compute loss and update weights
                loss = self.criterion(current_q_value, torch.tensor(target, dtype=torch.float32).to(device))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss.item()
    
    def train_with_deque(self, memory, batch_size=32, gamma=0.9):
        self.memory += memory
        return self.train_without_memory(batch_size=batch_size, gamma=gamma)
    
    def save(self):
        torch.save(self.state_dict(), 'tictactoe_model.pth')

            
model: TicTacToeNN = None
    

def getModel():
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
            train_model(model, episodes=1000, temperature=0.7)
    return model


# Training the model using Q-learning
def train_model(model: TicTacToeNN, episodes, epsilon=0.1, temperature=0.7):
    env = TicTacToe(keepMemory=True)

    valid=0
    invalid: int=0
    for episode in range(episodes):
        while not env.game_over:
            
            # Epsilon-greedy action selection with temperature-controlled softmax
            if random.random() < epsilon:
                action = random.choice(env.get_valid_moves())
            else:
                action = model_turn(model, env.board, temperature)

            if env.is_valid_move(action):
                valid +=1
            else:
                invalid +=1

            env.make_move(action)
            
            
        loss = env.trainAi()
        env.reset()
            
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Loss: {loss}")
            print(f"valid {valid}/{invalid}, {invalid/(valid+invalid)}")
            valid=0
            invalid=0
    
    # Save the trained model weights
    model.save()
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
        # # to make invalid moves possible
        # return np.argmax(val)
        # to only make valid moves possible
        valid_actions = [i for i in range(9) if state[i] == 0]
        return valid_actions[np.argmax(val[valid_actions])]
    else:
        # # to make invalid moves possible
        # return np.random.choice(9, p=val)
        # to only make valid moves possible
        valid_actions = [i for i in range(9) if state[i] == 0]
        # all valid actions are 0 so just return first valid action
        if not np.any(val[valid_actions]):
            return np.random.choice(valid_actions)
        prop = val[valid_actions] / np.sum(val[valid_actions])
        return np.random.choice(valid_actions, p=prop)
    
def get_matrix(state, temperature):
    return  get_matrix_with_model(getModel(), state, temperature)

def move(game: TicTacToe, temperature):
    if not game.game_over:
        for i in range(100):
            move = model_turn(getModel(), game.board, temperature)
            print(f"ai wants to move to {move}")
            if game.is_valid_move(move):
                return game.make_move(move)
        print(get_matrix(game.board, temperature))
        raise Exception("ai choose only invalid moves for 100 times")


