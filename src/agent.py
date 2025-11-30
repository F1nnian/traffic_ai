import numpy as np
from src.config import ALPHA, GAMMA, EPSILON_START


class QLearningAgent:
    """
    Q-learning agent for traffic light control.
    Uses a Q-table with shape (phase, bucketNS, bucketEW, action).
    """

    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START):
        # Q-table dimensions:
        # phase: 0 or 1                  (2)
        # bucketNS: 0,1,2,3              (4)
        # bucketEW: 0,1,2,3              (4)
        # action: stay or switch         (2)
        self.q = np.zeros((2, 4, 4, 2))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # -----------------------------------------------------
    # ACTION SELECTION (ε-greedy)
    # -----------------------------------------------------
    def act(self, state):
        """
        Select an action using epsilon-greedy strategy.
        state = (phase, bucketNS, bucketEW)
        returns: action 0 or 1
        """
        phase, b_ns, b_ew = state

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(2)  # random action (0 or 1)

        # Exploitation
        return int(np.argmax(self.q[phase, b_ns, b_ew]))

    # -----------------------------------------------------
    # Q-LEARNING UPDATE
    # -----------------------------------------------------
    def update(self, state, action, reward, next_state):
        """
        Update Q-table using the Bellman equation.
        """
        phase, b_ns, b_ew = state
        next_phase, nb_ns, nb_ew = next_state

        current_q = self.q[phase, b_ns, b_ew, action]
        max_next_q = np.max(self.q[next_phase, nb_ns, nb_ew])

        # TD target and TD error
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        # Update rule
        self.q[phase, b_ns, b_ew, action] += self.alpha * td_error

    # -----------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------
    def save(self, path):
        """Save the Q-table to a .npy file."""
        np.save(path, self.q)

    def load(self, path):
        """Load Q-table from a .npy file."""
        self.q = np.load(path)
