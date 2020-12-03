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
from distributed.actor import Actor
from distributed.learner import Learner
from distributed.lerner_per import LearnerPER
from distributed.param_server import ParamServer
from distributed.block_stacking_memory_server_per import BlockStackingMemoryServerPER
from utils.buffer import QLearningBuffer
from distributed.actor_state_server import ActorStateServer
from distributed.actor_monitor import ActorMonitor
from utils.Schedule import LinearSchedule

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')


@ray.remote(num_gpus=0.15)
class BlockStackingActor(Actor):
    pass


@ray.remote
class BlockStackingMemoryServer:
    def __init__(self, size):
        self.size = size
        self.storage = QLearningBuffer(size)

    def get_size(self):
        return len(self.storage)

    def add(self, item_list):
        [self.storage.add(t) for t in item_list]

    def sample(self, batch_size):
        return self.storage.sample(batch_size)


@ray.remote(num_gpus=0.15)
class BlockStackingParamServer(ParamServer):
    pass


# @ray.remote(num_gpus=0.15)
class BlockStackingLearner(Learner):
    def __init__(self, learn_params, env_params, param_server_remote, memory_server_remote):
        super().__init__(learn_params, env_params, param_server_remote, memory_server_remote)
        self.schedule = LinearSchedule(0.5, 0.01, self.epochs / 3)

    def eval_policy(self, episode):
        old_eps = self.agent.eps
        returns = []
        self.agent.eps = 0
        for _ in range(episode):
            G = 0
            rewards = []
            obs = self.test_env.reset()
            for _ in range(self.env_params['max_episode_time_steps']):
                action = self.agent.get_action(obs)
                next_obs, reward, done, _ = self.test_env.step(action)
                rewards.append(reward.item())
                if done.item():
                    for r in reversed(rewards):
                        G = r + self.agent.gamma * G
                    break
                else:
                    obs = next_obs
            returns.append(G)

        self.agent.eps = old_eps
        return returns


@ray.remote(num_gpus=0.15)
class BlockStackingLearnerPER(LearnerPER, BlockStackingLearner):
    def eval_policy(self, episode):
        return BlockStackingLearner.eval_policy(self, episode)


if __name__ == '__main__':
    ray.init(num_gpus=2)  # init the ray
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
                  'robot': 'kuka', 'workspace_check': 'point', 'in_hand_mode': 'raw', 'heightmap_resolution': heightmap_resolution}

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
        'crash_prob': 0.001,
        'report_alive_t': 0.1
    }

    # initialize parameters for training
    train_params = {
        'agent': None,
        'memory_size': 100000,
        'batch_size': 32,
        'episode_time_steps': 10,
        'epochs': 5000,
        'lr': 5e-5,
        'update_target_freq': 100,
        'start_train_memory_size': 100,
        'syn_param_freq': 10,
        'worker_num': 5,
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
    # create the actors
    actor_agents = []
    for i in range(train_params['worker_num']):
        actor_agents.append(DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'],
                                     agent_params['gamma']))
    actor_monitor = ActorMonitor(train_params['worker_num'], BlockStackingActor, actor_agents, agent_params, env_params,
                                 remote_param_server, remote_memory_server, remote_actor_state_server, actor_restart_t=5)

    test_returns = []
    losses = []
    while ray.get(remote_memory_server.get_size.remote()) < train_params['start_train_memory_size']:
        continue

    # start training
    pbar = tqdm.trange(train_params['epochs'])
    G = 0
    t0 = time.time()
    testing_freq = 5*60
    # testing_freq = 10
    for t in range(train_params['epochs']):
        actor_monitor.check_and_restart_actors()
        # sample a batch data
        loss = ray.get(remote_learner.update.remote())
        losses.append(loss)
        ray.get(remote_learner.sync_param_server.remote())

        # update the target policy
        if not np.mod(t, train_params['update_target_freq']):
            # update the target network
            ray.get(remote_learner.update_target.remote())

        if time.time() - t0 > (len(test_returns) + 1) * testing_freq:
            # if not np.mod(t, 200):
            G = ray.get(remote_learner.eval_policy.remote(100))
            test_returns.append(G)

        # print information
        pbar.set_description(
            f'Step: {t} |'
            f'Loss: {np.mean(losses[-10:]) if losses else 0:.3f} |'
            f'Test return: {np.mean(test_returns[-5:]) if test_returns else 0:.3f} | '
            f'Actor return: {ray.get(remote_actor_state_server.get_avg_return.remote()):.3f} | '
            # f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
        )
        pbar.update()
    np.save("./3s_per_10w_parallel.npy", test_returns)
    agent.fcn.load_state_dict(ray.get(remote_param_server.get_latest_model_params.remote()))
    agent.saveModel('./model')

    ray.wait(actor_monitor.actor_processes)

