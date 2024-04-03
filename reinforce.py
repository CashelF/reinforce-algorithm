from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.alpha = alpha
        self.network = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=alpha)


    def forward(self, states, return_prob=False):
        states = torch.tensor(states, dtype=torch.float32)
        x = self.network(states)
        x_prob = F.softmax(x, dim=-1)
        action = torch.distributions.Multinomial(1, x_prob).sample().argmax().item()
        return x_prob if return_prob else action


    def update(self, state, action_taken, gamma_t, delta):
        """
        states: states
        action_taken: action_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        action_probs = self.forward(state, return_prob=True)
        log_prob = torch.log(action_probs[action_taken])
        loss = -(log_prob * delta * gamma_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    There is no need to change this class.
    """
    def __init__(self,b):
        self.b = b
        
    def __call__(self, states):
        return self.forward(states)
        
    def forward(self, states) -> float:
        return self.b

    def update(self, states, G):
        pass

class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        self.state_dims = state_dims
        self.alpha = alpha
        self.network = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=alpha)

    def forward(self, states) -> float:
        states = torch.tensor(states, dtype=torch.float32)
        x = self.network(states)
        return x.squeeze() # TODO: is this right? should i squeeze it?


    def update(self, state, G):
        loss = F.mse_loss(self(state), G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:VApproximationWithNN) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G_0_list = []
    for episode in range(num_episodes):
        states = []
        actions_taken = []
        rewards = []
        state = env.reset()
        done = False
        while not done:
            action = pi(state, return_prob=False)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions_taken.append(action)
            rewards.append(reward)
            state = next_state

        G_0 = torch.tensor(0.0)
        gamma_t = 1
        for t in range(len(states)-1, -1, -1):
            G_0 = rewards[t] + gamma * G_0
            delta = G_0 - V(states[t])
            pi.update(states[t], actions_taken[t], gamma_t, delta)
            V.update(states[t], G_0)
            gamma_t *= gamma

        G_0_list.append(G_0.item())

    return G_0_list
