# Reinforcement-Learning-based-Agents-for-Text-Flappy-Bird
## Introduction
The goal of this assignment is to apply reinforcement learning methods to a simple game called Text Flappy Bird (TFB). TFB is a variation of the well-known Flappy Bird game in which the player is represented by a simple unit-element character. The environment for TFB is already implemented and can be found here. We will be using two RL agents, Q-learning and Sarsa, to train an agent for the TFB environment.

3# Agent Selection
We chose the Q-learning and Sarsa agents for this task because they are both model-free and can learn from experience. Additionally, they are both relatively simple to implement and have been shown to perform well on a variety of tasks. We considered other RL algorithms, such as Deep Q-Networks and Policy Gradient methods, but chose Q-learning and Sarsa because they are simpler to implement and can be trained relatively quickly on the TFB environment. Moreover, we also tried the Monte-Carlo method but the Q-learning and Sarsa provided better results.

### The Q-learning Agent
Q-learning is a model-free reinforcement learning algorithm that learns the optimal action-value function Q(s, a) by iteratively updating its estimate based on the Bellman equation:
Q(st, at) ← Q(st, at) + α · (rt+1 + γ · max a Q(st+1, a) − Q(st, at))
where s and a are the current state and action, s′ is the next state, r is the immediate reward, α is the learning rate, and γ is the discount factor.

### The Sarsa Agent
Sarsa is another model-free reinforcement learning algorithm that learns the optimal policy by iteratively updating its estimate based on the Bellman equation:
Q(st, at) ← Q(st, at) + α · (rt+1 + γ · Q(st+1, at+1) − Q(st, at))

where s and a are the current state and action, s′ is the next state, r is the immediate reward, α is the learning rate, and γ is the discount factor.

## Agent Comparison
The Q-learning and Sarsa agents are different in several ways. The Q-learning agent is more sensitive to the learning rate and the discount factor, while the Sarsa agent is more sensitive to the exploration-exploitation trade-off. The Q-learning agent tends to converge faster than the Sarsa agent, but it can be less stable. The Sarsa agent is less sensitive to rewards and can handle delayed rewards better than the Q-learning agent. For TFB, we will compare the performance of the Q-learning and Sarsa agents in further sections.

## State-Value Function Plots
<img src="https://user-images.githubusercontent.com/127759119/229242525-5d093f50-4a8e-44f3-aa82-9cbf47e8c48d.png" alt="alt text" width="500"/>


## Parameter Sweep
We swept step-size, epsilon (exploration rate), and discount factor for Q-learning and Sarsa agents to find the best hyperparameters. Over 10000 episodes, we averaged scores and rewards for each parameter. Step-size = 0.2, epsilon = 0.1 were optimal for both agents. Q-learning optimal discount value was 0.85 and Sarsa’s 0.
<img src="https://user-images.githubusercontent.com/127759119/229242567-cd75e3ee-3dd2-49a6-b61c-355f1ed78455.png" alt="alt text" width="500"/>


## Performance Comparison
To compare the performance of the two agents with a baseline, we also included a random agent that chooses
actions randomly from the action sample space. We ran each agent for 10000 episodes, and plotted the sum
of rewards and the sum of scores per run for each agent. The Q-learning and Sarsa agents outperformed the
random agent, with Q-learning showing slightly better performance. The Figures 3 & 4 shows the inital and
final results respectively.
<img src="https://user-images.githubusercontent.com/127759119/229242619-81c92815-0ff8-41fe-b95f-c4461e0994aa.png" alt="alt text" width="500"/>
<img src="https://user-images.githubusercontent.com/127759119/229242632-963fb6cb-908e-418b-bc94-1fdad9aa9f03.png" alt="alt text" width="500"/>
