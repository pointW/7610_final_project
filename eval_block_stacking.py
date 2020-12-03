import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch

import numpy as np
import matplotlib.pyplot as plt

from env.block_stacking.env_wrapper import BlockStackingEnv
from agent.block_stacking.dqn_wrapper import DQNBlock


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')


def eval():
    workspace = np.asarray([[0.35, 0.65],
                            [-0.15, 0.15],
                            [0, 0.50]])
    workspace_size = workspace[0][1] - workspace[0][0]
    heightmap_size = 90
    heightmap_resolution = workspace_size / heightmap_size

    # init the params
    env_config = {'simulator': 'pybullet', 'env': 'block_stacking', 'workspace': workspace, 'max_steps': 10,
                  'obs_size': heightmap_size, 'fast_mode': True, 'action_sequence': 'xyp', 'render': True,
                  'num_objects': 3, 'random_orientation': False, 'reward_type': 'sparse', 'simulate_grasp': True,
                  'robot': 'kuka', 'workspace_check': 'point', 'in_hand_mode': 'raw',
                  'heightmap_resolution': heightmap_resolution}

    # init the agent parameters
    agent_params = {
        'device': 'cuda',
        'lr': 5e-5,
        'gamma': 0.9,
        'crash_prob': 0,
        'report_alive_t': 0.1
    }

    # initialize parameters for training
    train_params = {
        'agent': None,
        'memory_size': 100000,
        'batch_size': 32,
        'episode_time_steps': 10,
        'epochs': 10000,
        'lr': 5e-5,
        'update_target_freq': 100,
        'start_train_memory_size': 100,
        'syn_param_freq': 10,
        'worker_num': 1,
        'per_alpha': 0.6,
        'per_beta': 0.4
    }

    plt.style.use('default')
    env = BlockStackingEnv(env_config)
    agent = DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])

    agent.loadModel('./model')
    agent.eval()
    obs = env.reset()
    test_episode = 1000
    total = 0
    s = 0
    pbar = tqdm(total=test_episode)
    while total < 1000:
        agent.eps = 0
        action = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
        else:
            obs = next_obs
        s += reward.sum().int().item()

        if done.sum():
            total += done.sum().int().item()

        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total, float(s) / total if total != 0 else 0)
        )
        pbar.update(done.sum().int().item())

if __name__ == '__main__':
    eval()