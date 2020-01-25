import torch
import numpy as np
from all.logging import DummyWriter
from ._agent import Agent


class FQF(Agent):
    """
    Fully Parameterized Quantile Function (FQF).

    Args:
        q_dist (QDist): Approximation of the Q distribution.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        eps (float): Stability parameter for computing the loss function.
        exploration (float): The probability of choosing a random action.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    """

    def __init__(
            self,
            fraction_proposer,
            quantile_network,
            replay_buffer,
            discount_factor=0.99,
            eps=1e-5,
            exploration=0.02,
            minibatch_size=32,
            replay_start_size=5000,
            update_frequency=1,
            writer=DummyWriter(),
    ):
        # objects
        self.fraction_proposer = fraction_proposer
        self.quantile_network = quantile_network
        self.replay_buffer = replay_buffer
        self.writer = writer
        # hyperparameters
        self.eps = eps
        self.exploration = exploration
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

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
        return (
            len(self.replay_buffer) < self.replay_start_size
            or np.random.rand() < self.exploration
        )

    def _best_action(self, state):
        proposed_fractions = self.fraction_proposer.eval(state)
        quantile_sizes = proposed_fractions[:, :, 1:] - proposed_fractions[:, 0:-1]
        quantile_centers = (proposed_fractions[:, :, 1:] + proposed_fractions[:, :, 0:-1]) / 2
        quantile_means = self.quantile_network.eval(quantile_centers, state)
        q = torch.sum(quantile_sizes * quantile_means, dim=2)
        return torch.argmax(q, dim=1)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            states, actions, rewards, next_states, _ = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            proposed_fractions = self.fraction_proposer(states, actions)
            quantile_centers = (proposed_fractions[:, 1:] + proposed_fractions[:, 0:-1]) / 2
            quantile_means = self.quantile_network(quantile_centers, states, actions)
            # compute targets
            next_proposed_fractions = self.fraction_proposer.target(next_states)
            next_quantile_sizes = next_proposed_fractions[:, :, 1:] - next_proposed_fractions[:, 0:-1]
            next_quantile_centers = (next_proposed_fractions[:, :, 1:] + next_proposed_fractions[:, :, 0:-1]) / 2
            next_quantile_means = self.quantile_network.target(next_quantile_centers, next_states)
            q = torch.sum(next_quantile_sizes * next_quantile_means, dim=2)
            next_actions = torch.argmax(q, dim=1)
            target_quantile_means = next_quantile_means[torch.arange(self.minibatch_size), next_actions]
            # compute loss
            quantile_td_error = rewards + self.discount_factor * target_quantile_means - quantile_means
            loss = self._huber_quantile_regression_loss(proposed_fractions, quantile_td_error)
            # compute dw/dtau
            # TODO
            dwdt = None
            # backward pass
            proposed_fractions.backward(dwdt)
            self.fraction_proposer.step()
            self.quantile_network.reinforce(loss)

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0

    def _huber_quantile_regression_loss(self, proposed_fractions, quantile_td_error):
        kappa = 1
        huber_loss_case_one = (quantile_td_error <= kappa).float() * 0.5 * quantile_td_error ** 2
        huber_loss_case_two = (quantile_td_error > kappa).float() * kappa * (quantile_td_error.abs() - 0.5 * kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
        return torch.abs(proposed_fractions - (quantile_td_error < 0).float()) * huber_loss / kappa
