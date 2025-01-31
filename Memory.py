class Memory:
    def __init__(self, prev_state, action, reward, next_state, done):
        self.prev_state = prev_state
        self.action = action
        self.reward = reward
        self.next_state = next_state	
        self.done = done
        
    def __str__(self):
        return str(self.prev_state) + "\n" + str(self.next_state) + "\n" \
            + str(self.action) + "\t" + str(self.reward) + "\t" + str(self.done)
    
    def to_tupel(self):
        return (self.prev_state, self.action, self.reward, self.next_state, self.done)
        