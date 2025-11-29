import numpy as np
import copy

from tensordict import TensorDict
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from agent_nn import AgentNN


class Agent:
    def __init__(self,
        network,
        num_actions,
        lr,
        gamma,
        epsilon,
        eps_decay,
        eps_min,
        replay_buffer_capacity,
        batch_size,
        sync_network_rate
    ):
        # Action count
        self.num_actions = num_actions

        # Counters
        self.learn_step_counter = 0
        self.episode_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Networks
        online_net = copy.deepcopy(network)
        target_net = copy.deepcopy(network)
        self.online_network = AgentNN(online_net, freeze=False)
        self.target_network = AgentNN(target_net, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.SmoothL1Loss()

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        # Choose random action with probability of epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Convert LazyFrames to a float tensor, add batch dimension, and move to the network's device.
        observation = (
            torch.tensor(np.array(observation), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.online_network.device)
        )

        # Retun the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        # Decay epsilon esnuring it is always higher than eps_min
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        # Store experience tuples
        self.replay_buffer.add(
            TensorDict(
                {
                    "state": torch.tensor(np.array(state), dtype=torch.float32),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
                    "done": torch.tensor(done),
                },
                batch_size=[],
            )
        )

    def sync_networks(self):
        # Update target network to match online network periodically
        if self.learn_step_counter % self.sync_network_rate == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        # If replay buffer not large enough, return
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Try to sync networks
        self.sync_networks()

        # Reset gradients
        self.optimizer.zero_grad()

        # Sample experiences from replay buffer
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        # Q-values predicted by the online network for the chosen actions
        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # Target Q-values computed from rewards + discounted max next-state Q-values from the target network
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        # Gradient descent
        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Increment and decay epsilon
        self.learn_step_counter += 1
        self.decay_epsilon()