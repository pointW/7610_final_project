import ray
import numpy as np
import gym
import os
import tqdm
import collections
import time

from agent.block_stacking.dqn_wrapper import DQNBlock
from network.models import ResUCatShared
from env.block_stacking.env_wrapper import BlockStackingEnv
from utils.ExperienceReplay import ReplayBuffer
from distributed.actor import Actor
from distributed.learner import Learner
from distributed.memory_server import MemoryServer
from distributed.param_server import ParamServer
from utils.Schedule import LinearSchedule
from utils.buffer import QLearningBuffer

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')


@ray.remote(num_gpus=0.25)
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


@ray.remote(num_gpus=0.25)
class BlockStackingParamServer(ParamServer):
    pass


@ray.remote(num_gpus=0.25)
class BlockStackingLearner(Learner):
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
        'total_time_steps': 10000,
        'env_fn': env_fn
    }

    # init the agent parameters
    agent_params = {
        'device': 'cuda',
        'lr': 5e-5,
        'gamma': 0.9,
    }

    # initialize parameters for training
    train_params = {
        'agent': None,
        'memory_size': 50000,
        'batch_size': 32,
        'episode_time_steps': 10,
        'epochs': 10000,
        'lr': 5e-5,
        'update_target_freq': 100,
        'update_policy_freq': 10,
        'start_train_memory_size': 100,
        'syn_param_freq': 10,
        'worker_num': 5
    }

    # create the remote memory server
    remote_memory_server = BlockStackingMemoryServer.remote(train_params['memory_size'])
    # create the agent
    agent = DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])
    train_params['agent'] = agent
    # agent_params['agent_model'] = agent
    # create the remote parameter server
    remote_param_server = BlockStackingParamServer.remote(agent.behavior_policy_net.state_dict())
    # create the remote learner server
    remote_learner = BlockStackingLearner.remote(train_params, env_params, remote_param_server, remote_memory_server)

    # create the actors
    # actor_num = 5
    actors = []
    for i in range(train_params['worker_num']):
        agent_params['agent_id'] = i
        agent_params['agent_model'] = DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])
        actors.append(BlockStackingActor.remote(agent_params, env_params, remote_param_server, remote_memory_server))

    processes = []
    for actor in actors:
        processes.append(actor)

    processes_running = [p.run.remote() for p in processes]

    test_returns = []
    losses = []
    while ray.get(remote_memory_server.get_size.remote()) < train_params['start_train_memory_size']:
        continue

    # start training
    pbar = tqdm.trange(train_params['epochs'])
    G = 0
    t0 = time.time()
    testing_freq = 2*60
    # testing_freq = 10
    for t in range(train_params['epochs']):
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
            G = np.mean(ray.get(remote_learner.eval_policy.remote(20)))
            test_returns.append(G)

        # print information
        pbar.set_description(
            f'Step: {t} |'
            f'Loss: {np.mean(losses[-10:]) if losses else 0:.4f} |'
            f'G: {np.mean(test_returns[-5:]) if test_returns else 0:.2f} | '
            # f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
        )
        pbar.update()
    np.save("./parallel_returns.npy", test_returns)

    ray.wait(processes_running)

    # processes = []
    # for actor in actors:
    #     processes.append(actor)
    # processes_running = [p.run.remote() for p in processes]
    # ray.wait(processes_running)

