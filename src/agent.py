import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_dim, action_dim, alpha=0.1, gamma=0.99, epsilon=1.0):
        """
        Initializes the Q-Learning Agent.
        
        Args:
            state_dim (tuple): Shape of the state space (e.g., (2, 4, 4)).
            action_dim (int): Number of actions (e.g., 2).
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration probability.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-Table with zeros
        # We use a numpy array of shape (*state_dim, action_dim)
        # Example: (2, 4, 4, 2)
        self.q_table = np.zeros(state_dim + (action_dim,))

    def act(self, state, explore=True):
        """
        Chooses an action based on the state.
        
        Args:
            state (tuple): Current state indices, e.g., (0, 2, 3).
            explore (bool): Whether to use epsilon-greedy exploration.
            
        Returns:
            int: The chosen action (0 or 1).
        """
        # SKELETON LOGIC:
        # If explore is True, we pick random action sometimes.
        # Otherwise, we pick the best action from q_table.
        
        # For now, let's just return a random action so the code runs
        return random.choice([0, 1])

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-Table using the Bellman Equation.
        
        Args:
            state (tuple): Previous state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): New state.
            done (bool): Whether episode ended.
        """
        # TODO: Implement Q-Learning Math here
        # Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        pass

    def save(self, filepath):
        """Saves the Q-table to a file."""
        np.save(filepath, self.q_table)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Loads the Q-table from a file."""
        try:
            self.q_table = np.load(filepath)
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print("No saved model found, starting from scratch.")
