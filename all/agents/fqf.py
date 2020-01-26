import numpy as np
import torch
from all.logging import DummyWriter
from ._agent import Agent


class FQF(Agent):
    """
    Fully Parameterized Quantile Function (FQF).
    Starting with IQN.

    Args:
        q_dist (QDist): Approximation of the Q distribution.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        exploration (float): The probability of choosing a random action.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    """

    def __init__(
            self,
            f,
            quantile_network,
            replay_buffer,
            discount_factor=0.99,
            exploration=0.02,
            huber_loss_threshold=1,
            minibatch_size=32,
            num_tau_samples=32,
            replay_start_size=5000,
            update_frequency=1,
            writer=DummyWriter(),
    ):
        # objects
        self.f = f
        self.quantile_network = quantile_network
        self.replay_buffer = replay_buffer
        self.writer = writer
        # hyperparameters
        self.exploration = exploration
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.huber_loss_threshold = huber_loss_threshold
        self.num_tau_samples = num_tau_samples
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0
        self._quantiles_act = (
            torch.linspace(0, 1, num_tau_samples).view((1, num_tau_samples)).to(self.quantile_network.device))
        self._quantiles_train = (torch.linspace(0, 1, num_tau_samples)
                                 .expand((minibatch_size, num_tau_samples))
                                 .to(self.quantile_network.device))

    def act(self, state, reward):
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _choose_action(self, state):
        if self._should_explore():
            return torch.randint(
                self.quantile_network.n_actions, (len(state),), device=self.quantile_network.device
            )
        return self._best_action(state)

    def _should_explore(self):
        return (np.random.rand() < self.exploration)
        # return (
        #     len(self.replay_buffer) < self.replay_start_size
        #     or np.random.rand() < self.exploration
        # )

    def _best_action(self, state):
        quantile_values = self.quantile_network.eval(self._quantiles_act, self.f.eval(state))
        q = torch.mean(quantile_values, dim=-1)
        return torch.argmax(q, dim=1)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            states, actions, rewards, next_states, _ = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            self._quantiles_train = torch.randn((self.minibatch_size, self.num_tau_samples), device=self.quantile_network.device)
            quantile_values = self.quantile_network(self._quantiles_train, self.f(states), actions)
            # compute targets
            next_quantile_values = self.quantile_network.target(self._quantiles_train, self.f.target(next_states))
            next_q = torch.mean(next_quantile_values, dim=-1)
            next_actions = torch.argmax(next_q, dim=1)
            next_quantile_values = next_quantile_values[torch.arange(self.minibatch_size), next_actions]
            target_quantile_values = rewards[:, None] + self.discount_factor * next_quantile_values
            # compute loss
            quantile_td_error = target_quantile_values[:, :, None] - quantile_values[:, None, :]
            loss = self._huber_quantile_regression_loss(quantile_td_error).mean()
            # backward pass
            self.quantile_network.reinforce(loss)
            self.f.reinforce()
            # debugging
            self.writer.add_loss('q_mean', quantile_values.detach().mean())

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0

    def _huber_quantile_regression_loss(self, quantile_td_error):
        k = self.huber_loss_threshold
        abs_quantile_td_error = torch.abs(quantile_td_error)
        huber_loss_case_one = (abs_quantile_td_error <= k).float() * 0.5 * quantile_td_error ** 2
        huber_loss_case_two = (abs_quantile_td_error > k).float() * k * (abs_quantile_td_error - 0.5 * k)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        return (torch.abs(self._quantiles_train[:, :, None] - (quantile_td_error < 0).float()) * huber_loss) / k
