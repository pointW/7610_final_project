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
from utils.Schedule import LinearSchedule
from utils.buffer import QLearningBuffer

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')


def eval_policy(env, policy, episode, max_episode_time_steps, gamma=0.9):
    # env_test = BlockStackingEnv(env_params['env_config'])
    old_eps = policy.eps
    returns = []
    policy.eps = 0
    for _ in range(episode):
        G = 0
        rewards = []
        obs = env.reset()
        for _ in range(max_episode_time_steps):
            action = policy.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            rewards.append(reward.item())
            if done.item():
                for r in reversed(rewards):
                    G = r + gamma * G
                break
            else:
                obs = next_obs
        returns.append(G)

    policy.eps = old_eps
    return np.average(returns)


@ray.remote(num_gpus=0.25)
class Actor(object):
    def __init__(self, agent_params, env_params, remote_param_server, remote_memory_server):
        # store the IDs
        self.id = agent_params['agent_id']
        self.agent = agent_params['agent_model']

        # init remote servers
        self.remote_param_server = remote_param_server
        self.remote_memory_server = remote_memory_server

        # local experience replay buffer
        self.local_buffer = []
        self.buffer_size = env_params['max_episode_time_steps']

        # environment parameters
        self.env = BlockStackingEnv(env_params['env_config'])
        self.total_time_steps = env_params['total_time_steps']
        self.episode_time_steps = env_params['max_episode_time_steps']

        # running indicators
        self.current_step = 0

        self.rewards = []

        self.schedule = LinearSchedule(1, 0.01, self.total_time_steps / 2)

    def update_behavior_policy(self):
        self.agent.behavior_policy_net.load_state_dict(ray.get(self.remote_param_server.get_model_params.remote()))
        self.agent.behavior_policy_net.eval()
        self.current_step = ray.get(self.remote_param_server.get_step_params.remote())

    def send_data(self):
        # send the data to the memory server
        self.remote_memory_server.add.remote(self.local_buffer)
        # clear the local memory buffer
        self.local_buffer = []

    def run(self):
        # synchronize the parameters
        self.update_behavior_policy()
        # initialize the environment
        obs, rewards = self.env.reset(), []
        # start data collection
        while True:
            # compute the epsilon
            self.agent.eps = self.scheduled_eps
            # get the action
            action = self.agent.get_action(obs)
            # interaction with the environment
            next_obs, reward, done, _ = self.env.step(action)
            # record rewards
            rewards.append(reward)
            # add the local buffer
            self.local_buffer.append((obs, action, reward, next_obs, done))
            # check termination
            if done:
                # G = 0
                # for r in reversed(rewards):
                #     G = r + 0.9995 * G
                # print(f"Actor {self.id}: G = {G}, Eps = {self.scheduled_eps}")
                # reset environment
                obs, rewards = self.env.reset(), []
                # synchronize the behavior policy
                self.update_behavior_policy()
                # send data to remote memory buffer
                self.send_data()
            else:
                obs = next_obs


@ray.remote
class MemoryServer(object):
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
class ParamServer(object):
    def __init__(self, model_state_dict):
        self.param_model_state_dict = model_state_dict
        self.param_step = 0

    def sync_params(self, model_stat_dict, param_step):
        self.param_model_state_dict = model_stat_dict
        self.param_step = param_step

    def get_model_params(self):
        return self.param_model_state_dict

    def get_step_params(self):
        return self.param_step


@ray.remote(num_gpus=0.25)
class Learner(object):
    def __init__(self, learn_params, env_params, remote_param_server, remote_memory_server):
        # remote servers
        self.remote_memory_server = remote_memory_server
        self.remote_param_server = remote_param_server

        # learning params
        self.agent = learn_params['agent']
        self.test_env = BlockStackingEnv(env_params['env_config'])

        self.step = 0
        self.batch_size = learn_params['batch_size']
        self.start_train_step = learn_params['start_train_step']
        self.total_time_steps = learn_params['total_time_steps']
        self.update_target_freq = learn_params['update_target_freq']
        self.update_policy_freq = learn_params['update_policy_freq']

        # server synchronize frequency
        self.sync_param_server_freq = learn_params['syn_param_freq']

        self.returns = []
        self.steps = []

    def get_model_params(self):
        return self.param_model

    def get_step_params(self):
        return self.param_step

    def sync_param_server(self):
        self.remote_param_server.sync_params.remote(self.agent.behavior_policy_net.state_dict(), self.step)

    def set_step(self, step):
        self.step = step

    def update(self):
        batch_data = ray.get(self.remote_memory_server.sample.remote(self.batch_size))
        loss, td_error = self.agent.update_behavior_policy(batch_data)
        self.step += 1
        return loss

    def update_target(self):
        self.agent.update_target_policy()

    def eval_policy(self):
        return eval_policy(self.test_env, self.agent, 20, 10)


ray.init(num_gpus=2)  # init the ray

if __name__ == '__main__':
    workspace = np.asarray([[0.35, 0.65],
                            [-0.15, 0.15],
                            [0, 0.50]])
    workspace_size = workspace[0][1] - workspace[0][0]
    heightmap_size = 90
    heightmap_resolution = workspace_size / heightmap_size

    # init the params
    env_config = {'simulator': 'pybullet', 'env': 'block_stacking', 'workspace': workspace, 'max_steps': 10,
                  'obs_size': heightmap_size, 'fast_mode': True, 'action_sequence': 'xyp', 'render': False,
                  'num_objects': 4, 'random_orientation': False, 'reward_type': 'sparse', 'simulate_grasp': True,
                  'robot': 'kuka', 'workspace_check': 'point', 'in_hand_mode': 'raw', 'heightmap_resolution': heightmap_resolution}

    env_params = {
        'env_name': 'block_stacking',
        'max_episode_time_steps': 10,
        'total_time_steps': 10000,
        'env_config': env_config
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
        'total_time_steps': 10000,
        'lr': 5e-5,
        'update_target_freq': 100,
        'update_policy_freq': 10,
        'start_train_step': 100,
        'syn_param_freq': 10
    }

    # create the remote memory server
    remote_memory_server = MemoryServer.remote(train_params['memory_size'])
    # create the agent
    agent = DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])
    train_params['agent'] = agent
    # agent_params['agent_model'] = agent
    # create the remote parameter server
    remote_param_server = ParamServer.remote(agent.behavior_policy_net.state_dict())
    # create the remote learner server
    remote_learner = Learner.remote(train_params, env_params, remote_param_server, remote_memory_server)

    # create the actors
    actor_num = 5
    actors = []
    for i in range(actor_num):
        agent_params['agent_id'] = i
        agent_params['agent_model'] = DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])
        actors.append(Actor.remote(agent_params, env_params, remote_param_server, remote_memory_server))

    processes = []
    for actor in actors:
        processes.append(actor)

    processes_running = [p.run.remote() for p in processes]

    test_returns = []
    losses = []
    while ray.get(remote_memory_server.get_size.remote()) < train_params['start_train_step']:
        continue

    # start training
    pbar = tqdm.trange(train_params['total_time_steps'])
    G = 0
    t0 = time.time()
    testing_freq = 2*60
    for t in range(train_params['total_time_steps']):
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
            G = ray.get(remote_learner.eval_policy.remote())
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

