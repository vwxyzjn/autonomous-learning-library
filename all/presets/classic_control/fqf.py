from torch.optim import Adam
from all.agents import FQF
from all.approximation import FeatureNetwork, QuantileNetwork
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from .models import fc_relu_features, fc_policy_head


def fqf(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=1e-3,
        # Training settings
        minibatch_size=128,
        update_frequency=1,
        # Replay buffer settings
        replay_start_size=1000,
        replay_buffer_size=20000,
        # Exploration settings
        initial_exploration=1.00,
        final_exploration=0.02,
        final_exploration_frame=10000,
):
    """
    FQF classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr (float): Learning rate for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (int): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (int): Final probability of choosing a random action.
        atoms (int): The number of atoms in the categorical distribution used to represent
            the distributional value function.
        v_min (int): The expected return corresponding to the smallest atom.
        v_max (int): The expected return correspodning to the larget atom.
    """
    def _fqf(env, writer=DummyWriter()):
        feature_model = fc_relu_features(env, hidden=32).to(device)
        quantile_model = fc_policy_head(env, hidden=64).to(device)
        feature_optimizer = Adam(feature_model.parameters(), lr=lr)
        quantile_optimizer = Adam(quantile_model.parameters(), lr=lr)
        f = FeatureNetwork(feature_model, feature_optimizer, writer=writer)
        q = QuantileNetwork(quantile_model, quantile_optimizer, env.action_space.n, 32, writer=writer)

        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, device=device)

        return FQF(
            f,
            q,
            replay_buffer,
            exploration=LinearScheduler(
                initial_exploration,
                final_exploration,
                replay_start_size,
                final_exploration_frame,
                name="epsilon",
                writer=writer,
            ),
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
            num_tau_samples=32,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency,
            writer=writer
        )

    return _fqf

__all__ = ["fqf"]
