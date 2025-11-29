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

        # Convert LazyFrames -> numpy -> torch, normalize to [0, 1]
        obs = torch.from_numpy(np.asarray(observation)).to(self.online_network.device)
        obs = obs.float() / 255.0
        obs = obs.unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(obs)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        # Decay epsilon esnuring it is always higher than eps_min
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        # Store experience tuples
        self.replay_buffer.add(
            TensorDict(
                {
                    "state": torch.from_numpy(np.asarray(state)).to(torch.uint8),
                    "action": torch.tensor(action, dtype=torch.int64),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "next_state": torch.from_numpy(np.asarray(next_state)).to(torch.uint8),
                    "done": torch.tensor(done, dtype=torch.bool),
                },
                batch_size=[],
            )
        )

    def sync_networks(self):
        # Update target network to match online network periodically
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
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
        states = states.float() / 255.0
        next_states = next_states.float() / 255.0

        # Q-values predicted by the online network for the chosen actions
        q_values = self.online_network(states)
        batch_indices = torch.arange(self.batch_size, device=actions.device)
        predicted_q_values = q_values[batch_indices, actions.squeeze(-1)]

        # Target Q-values computed from rewards + discounted max next-state Q-values from the target network
        next_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())

        # Gradient descent
        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Increment and decay epsilon
        self.learn_step_counter += 1
        self.decay_epsilon()