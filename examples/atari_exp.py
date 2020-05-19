'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import run_experiment, plot_returns_100
# from all.presets.classic_control import c51
# from all.presets import atari
from all.environments import GymEnvironment
from all.environments.atari2 import Atari2Environment
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork, FixedTarget
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from all import nn
# from .models import nature_dqn
import torch
import numpy as np
torch.manual_seed(2)
torch.backends.cudnn.deterministic = True
np.random.seed(2)
def nature_dqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n)
    )
def dqn(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=1e-4,
        eps=1.5e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=4,
        target_update_frequency=1000,
        # Replay buffer settings
        replay_start_size=80000,
        replay_buffer_size=1000000,
        # Explicit exploration
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=4000000,
):
    """
    DQN Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (int): Initial probability of choosing a random action,
            decayed until final_exploration_frame.
        final_exploration (int): Final probability of choosing a random action.
        final_exploration_frame (int): The frame where the exploration decay stops.
    """
    def _dqn(env, writer=DummyWriter()):
        action_repeat = 4
        last_timestep = last_frame / action_repeat
        last_update = (last_timestep - replay_start_size) / update_frequency
        final_exploration_step = final_exploration_frame / action_repeat
        print(final_exploration_step)

        model = nature_dqn(env).to(device)
        print(list(model.children())[1].weight.sum())

        optimizer = Adam(
            model.parameters(),
            lr=lr,
            # eps=eps
        )

        q = QNetwork(
            model,
            optimizer,
            # scheduler=CosineAnnealingLR(optimizer, last_update),
            target=FixedTarget(target_update_frequency),
            writer=writer
        )

        policy = GreedyPolicy(
            q,
            env.action_space.n,
            epsilon=LinearScheduler(
                initial_exploration,
                final_exploration,
                0,
                final_exploration_step,
                name="epsilon",
                writer=writer
            )
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )

        return DQN(
                q,
                policy,
                replay_buffer,
                discount_factor=discount_factor,
                # loss=smooth_l1_loss,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
            )
    return _dqn


def main():
    device = 'cpu'
    timesteps = 1e7
    env = Atari2Environment('BreakoutNoFrameskip-v4', device=device)
    env.seed(2)
    run_experiment(
        [dqn(device=device)],
        [env],
        timesteps,
    )
    # plot_returns_100('runs', timesteps=timesteps)

if __name__ == "__main__":
    import wandb
    wandb.init(
        project="cleanrl",
        entity="cleanrl",
        sync_tensorboard=True,
        name="dqn_same_env",
        monitor_gym=True,
        save_code=True)
    main()


# from gym import envs
# from all.experiments import run_experiment
# from all.presets import atari
# from all.environments import AtariEnvironment

# agents = [
#     atari.c51()
# ]

# envs = [AtariEnvironment(env, device='cpu') for env in ['Pong']]

# run_experiment(agents, envs, 10e6)