# Architecture

## What we're building

At a high level, we are developing a reinforcement‑learning (RL) agent to control the traffic lights in a simple intersection in simulation. The agent will observe the state of the intersection (cars waiting, current phase) and decide whether to stay in the current phase or switch to the other phase. We will start with a Tabular Q-Learning approach on a single intersection and add complexity later. The core goal is to minimize the delay using a squared wait-time penalty. In a nutshell, our system is an RL environment plus an agent.

## Pieces of the code

- Simulator (`src/env.py`): generates cars, updates/tracks their positions and waiting times, and keeps track of which phase of the traffic light is active as well as changes phases.
- RL agent (`src/agent.py`): implements tabular Q‑learning or (later) neural approaches (such as DQN) to select actions and learn from the rewards returned by the simulator.
- Training loop (`src/train.py`): runs episodes, collects state–action–reward data and updates the agent. We might use a library like `stable‑baselines3` for advanced algorithms (e.g. PPO).
- Evaluation / plotting (`src/evaluate.py` and `src/visualize.py`): tests the trained agent against baseline controllers (such as fixed‑cycle schedules) and visualises metrics like waiting time, queue length and throughput.

## How things flow

1. Initialize the environment and agent.
2. For each episode:
   - Reset the environment and, if needed, reset the agent.
   - For each step:
     1. Observe the current state of the intersection.
     2. Choose an action (stay in the current phase or switch to the other phase).
     3. Apply the action in the environment; the environment returns the next state, a reward and a done flag.
     4. Update the agent (if training) and accumulate rewards.
   - Record results and adjust exploration parameters if necessary.
3. Evaluate performance: compute average waiting times, throughput and other metrics, then compare the RL controller to simple fixed‑cycle baselines.

## File structure

For now we plan to organise the code like this:

- `src/env.py` – the simulation logic and state representation.
- `src/agent.py` – the RL agent implementation.
- `src/train.py` – the training loop and hyperparameters.
- `src/evaluate.py` – scripts for testing and visualising results.
- `src/visualize.py` –> maybe use a jupyter notebook instead

This document is a work in progress.
