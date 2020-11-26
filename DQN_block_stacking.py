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
        os.environ["MKL_NUM_THREADS"] = "1"
        self.id = agent_params['agent_id']
        self.agent = agent_params['agent_model']
        self.remote_param_server = remote_param_server
        self.remote_memory_server = remote_memory_server

        # local experience replay buffer
        self.local_buffer = []
        self.buffer_size = 32

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

    def send_transition_to_memory_server(self):
        self.remote_memory_server.add.remote(self.local_buffer)
        self.local_buffer = []

    def single_rollout(self):
        # reset the domain
        obs = self.env.reset()
        rewards = []
        # perform one rollout
        for t in range(self.episode_time_steps):
            # get one action
            self.agent.eps = self.schedule.get_value(self.current_step)
            action = self.agent.get_action(obs)
            # perform one step
            next_obs, reward, done, _ = self.env.step(action)

            # save transitions
            self.local_buffer.append((obs, action, reward, next_obs, done))
            rewards.append(reward)

            if len(self.local_buffer) > self.buffer_size:
                # pass the transitions to buffer server
                self.send_transition_to_memory_server()

            if done:
                break
            else:
                obs = next_obs
        return rewards

    def run(self):
        # keep collecting data until total time steps used up
        episode_count = 0
        # pbar = tqdm.tqdm(total=self.total_time_steps)
        while self.current_step < self.total_time_steps:
            prev_step = self.current_step
            # perform rollouts
            # if not np.mod(episode_count, 10):
            self.send_transition_to_memory_server()
            self.update_behavior_policy()

            episode_rewards = self.single_rollout()
            self.rewards.append(np.sum(episode_rewards))
            episode_count += 1
            starting = max(0, len(self.rewards) - 1 - 1000)
            avg = np.sum(self.rewards[starting:]) / 1000
            # pbar.set_description('episode: {}, running reward: {}'.format(episode_count, avg))
            # pbar.update(self.current_step - prev_step)


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
        return eval_policy(self.test_env, self.agent, 10, 10)

    # def run(self):
    #     # check if the memory server contains enough data
    #     while ray.get(self.remote_memory_server.get_size.remote()) < self.start_train_step:
    #         continue
    #
    #     # start training
    #     pbar = tqdm.trange(self.total_time_steps)
    #     for t in range(self.total_time_steps):
    #         # increase the steps
    #         self.step = t
    #
    #         # update the behavior policy
    #         if not np.mod(self.step, self.update_policy_freq):
    #             # sample a batch data
    #             batch_data = ray.get(self.remote_memory_server.sample.remote(self.batch_size))
    #             self.agent.update_behavior_policy(batch_data)
    #             self.sync_param_server()
    #
    #         # update the target policy
    #         if not np.mod(self.step, self.update_target_freq):
    #             # update the target network
    #             self.agent.update_target_policy()
    #
    #         if not np.mod(self.step, 200):
    #             G = eval_policy(env_params, self.agent)
    #             self.returns.append(G)
    #             self.steps.append(t)
    #
    #         # print information
    #         pbar.set_description(
    #                 f'Step: {self.step} |'
    #                 f'G: {G:.2f} | '
    #                 f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
    #         )
    #     np.save("./w3_returns.npy", self.returns)


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
                  'num_objects': 3, 'random_orientation': False, 'reward_type': 'sparse', 'simulate_grasp': True,
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

    # create the actor
    agent_params['agent_id'] = 0
    agent_params['agent_model'] = DQNBlock(workspace, heightmap_resolution, agent_params['device'], agent_params['lr'], agent_params['gamma'])
    actor = Actor.remote(agent_params, env_params, remote_param_server, remote_memory_server)

    test_returns = []
    losses = []

    # start training
    pbar = tqdm.trange(train_params['total_time_steps'])
    G = 0
    t0 = time.time()
    testing_freq = 2*60
    t = 0
    while t < train_params['total_time_steps']:
        ray.get(actor.send_transition_to_memory_server.remote())
        ray.get(actor.update_behavior_policy.remote())
        ray.get(actor.single_rollout.remote())
        if ray.get(remote_memory_server.get_size.remote()) < train_params['start_train_step']:
            continue
        # sample a batch data
        for _ in range(10):
            loss = ray.get(remote_learner.update.remote())
            losses.append(loss)
            ray.get(remote_learner.sync_param_server.remote())
            t += 1

        # update the target policy
        if not np.mod(t, train_params['update_target_freq']):
            # update the target network
            ray.get(remote_learner.update_target.remote())

        if time.time() - t0 > (len(test_returns) + 1) * testing_freq:
            G = ray.get(remote_learner.eval_policy.remote())
            test_returns.append(G)

        # print information
        pbar.set_description(
            f'Step: {t} |'
            f'Loss: {np.mean(losses[-100:]) if losses else 0:.4f} |'
            f'G: {np.mean(test_returns[-100:]) if test_returns else 0:.2f} | '
            # f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
        )
        pbar.update(t - pbar.n)
    np.save("./single_returns.npy", test_returns)


