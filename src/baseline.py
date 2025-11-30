"""
This file implements a Fixed-Time Controller.
It ignores the state (traffic queues) and simply switches the light
every 'cycle_duration' steps.

We use this to benchmark our AI. If the AI cannot beat this, 
the AI is not learning correctly.
"""

class FixedTimeAgent:
    def __init__(self, cycle_duration=30):
        """
        Args:
            cycle_duration (int): How many steps to keep the light green 
                                  before switching.
        """
        self.cycle_duration = cycle_duration
        self.step_counter = 0

    def act(self, state):
        """
        Decides an action based on time, ignoring traffic density.
        
        Args:
            state (tuple): The current state from the environment 
                           (Phase, BucketNS, BucketEW). We ignore this!
        
        Returns:
            int: 0 (Stay) or 1 (Switch)
        """
        # Increment our internal timer
        self.step_counter += 1

        # Check if time is up
        if self.step_counter >= self.cycle_duration:
            # Time to switch! Reset timer and return 1
            self.step_counter = 0
            return 1
        else:
            # Keep the light green. Return 0
            return 0