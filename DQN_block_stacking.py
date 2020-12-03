import ray
import numpy as np
import gym
import os
import tqdm
import collections
import time

from agent.block_stacking.dqn_wrapper import DQNBlock
from agent.block_stacking.dqn_per import DQNBlockPER
from env.block_stacking.env_wrapper import BlockStackingEnv
from utils.Schedule import LinearSchedule
from utils.buffer import QLearningBuffer
from distributed_DQN_ray_block_stacking import BlockStackingActor, BlockStackingMemoryServer, BlockStackingParamServer, BlockStackingLearner, BlockStackingLearnerPER
from distributed.block_stacking_memory_server_per import BlockStackingMemoryServerPER
from distributed.actor_state_server import ActorStateServer

if __name__ == '__main__':
    ray.init(num_gpus=1)  # init the ray
    workspace = np.asarray([[0.35, 0.65],
                            [-0.15, 0.15],
                            [0, 0.50]])
    workspace_size = workspace[0][1] - workspace[0][0]
    heightmap_size = 90
    heightmap_resolution = workspace_size / heightmap_size

    # init the params
    env_config = {'simulator': 'pybullet', 'env': 'block_stacking', 'workspace': workspace, 'max_steps': 10,
                  'obs_size': heightmap_size, 'fast_mode': True, 'action_sequence': 'xyp', 'render': False,
                  'num_objects': 3, 'random_orientation': False, 'reward_type': 'sparse', 'simulate_grasp': True,
                  'robot': 'kuka', 'workspace_check': 'point', 'in_hand_mode': 'raw',
                  'heightmap_resolution': heightmap_resolution}


    def env_fn():
        return BlockStackingEnv(env_config)


    env_params = {
        'env_name': 'block_stacking',
        'max_episode_time_steps': 10,
        'env_fn': env_fn
    }

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
    PER = True

    # create the remote memory server
    if PER:
        remote_memory_server = ray.remote(BlockStackingMemoryServerPER).remote(train_params['memory_size'], train_params['per_alpha'])
    else:
        remote_memory_server = BlockStackingMemoryServer.remote(train_params['memory_size'])
    # create the agent
    if PER:
        agent = DQNBlockPER(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])
    else:
        agent = DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])
    train_params['agent'] = agent
    # agent_params['agent_model'] = agent
    # create the remote parameter server
    remote_param_server = BlockStackingParamServer.remote(agent.behavior_policy_net.state_dict(), train_params['epochs'])
    # create the remote learner server
    if PER:
        remote_learner = BlockStackingLearnerPER.remote(train_params, env_params, remote_param_server, remote_memory_server)
    else:
        remote_learner = BlockStackingLearner.remote(train_params, env_params, remote_param_server, remote_memory_server)
    remote_actor_state_server = ray.remote(ActorStateServer).remote(train_params['worker_num'])

    test_returns = []
    losses = []

    # start training
    pbar = tqdm.trange(train_params['epochs'])
    G = 0
    t0 = time.time()
    testing_freq = 5*60
    t = 0
    env = env_fn()
    obs, rewards = env.reset(), []
    for t in range(train_params['epochs']):
        agent.behavior_policy_net.load_state_dict(ray.get(remote_param_server.get_latest_model_params.remote()))
        agent.eps = ray.get(remote_param_server.get_scheduled_eps.remote())
        action = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward.item())
        remote_memory_server.add.remote([(obs, action, reward, next_obs, done)])
        if done:
            G = 0
            for r in reversed(rewards):
                G = r + agent.gamma * G
            remote_actor_state_server.add_return.remote(G)
            obs, rewards = env.reset(), []
        else:
            obs = next_obs
        if ray.get(remote_memory_server.get_size.remote()) < train_params['start_train_memory_size']:
            continue
        loss = ray.get(remote_learner.update.remote())
        losses.append(loss)
        ray.get(remote_learner.sync_param_server.remote())

        # update the target policy
        if not np.mod(t, train_params['update_target_freq']):
            # update the target network
            ray.get(remote_learner.update_target.remote())

        if time.time() - t0 > (len(test_returns) + 1) * testing_freq:
            G = ray.get(remote_learner.eval_policy.remote(20))
            test_returns.append(G)

        # print information
        pbar.set_description(
            f'Step: {t} |'
            f'Loss: {np.mean(losses[-10:]) if losses else 0:.4f} |'
            f'G: {np.mean(test_returns[-5:]) if test_returns else 0:.2f} | '
            f'Actor return: {ray.get(remote_actor_state_server.get_avg_return.remote()):.3f} | '
            # f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
        )
        pbar.update(t - pbar.n)
    np.save("./3s_per_single.npy", test_returns)


