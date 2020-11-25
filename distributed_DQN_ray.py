import ray
import numpy as np
import gym
import os
import tqdm
import copy

from agent.DQNAgent import DQNAgent
from utils.ExperienceReplay import ReplayBuffer
from utils.Schedule import LinearSchedule


def eval_policy(env, policy):
    env_test = gym.make(env['env_name'])
    obs, rewards = env_test.reset(), []
    old_eps = policy.eps

    policy.eps = 0
    for t in range(env['max_episode_time_steps']):
        action = policy.get_action(obs)
        next_obs, reward, done, _ = env_test.step(action)
        rewards.append(reward)
        if done:
            G = 0
            for r in reversed(rewards):
                G = r + 0.9995 * G
            break
        else:
            obs = next_obs

    policy.eps = old_eps
    return G


@ray.remote
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
        self.buffer_size = 200

        # environment parameters
        self.env = gym.make(env_params['env_name'])
        self.total_time_steps = env_params['total_time_steps']
        self.episode_time_steps = env_params['max_episode_time_steps']

        # running indicators
        self.current_step = 0
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

            if done:
                # G = 0
                # for r in reversed(rewards):
                #     G = r + 0.9995 * G
                # print(f"Step{self.current_step} Actor: {G}")
                break
            else:
                obs = next_obs

    def run(self):
        # keep collecting data until total time steps used up
        episode_count = 0
        while self.current_step < self.total_time_steps:
            # perform rollouts
            # if not np.mod(episode_count, 10):
            self.send_transition_to_memory_server()
            self.update_behavior_policy()

            self.single_rollout()
            episode_count += 1


@ray.remote
class MemoryServer(object):
    def __init__(self, size):
        self.size = size
        self.storage = ReplayBuffer(size)

    def get_size(self):
        return len(self.storage)

    def add(self, item_list):
        for item in item_list:
            obs, act, reward, next_obs, d = item
            self.storage.add(obs, act, reward, next_obs, d)

    def sample(self, batch_size):
        return self.storage.sample_batch(batch_size)


@ray.remote
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


@ray.remote
class Learner(object):
    def __init__(self, learn_params, remote_param_server, remote_memory_server):
        # remote servers
        self.remote_memory_server = remote_memory_server
        self.remote_param_server = remote_param_server

        # learning params
        self.agent = learn_params['agent']
        self.step = 0
        self.batch_size = learn_params['batch_size']
        self.start_train_step = learn_params['start_train_step']
        self.total_time_steps = learn_params['total_time_steps']
        self.update_target_freq = learn_params['update_target_freq']
        self.update_policy_freq = learn_params['update_policy_freq']

        # save results
        self.returns = []
        self.steps = []

    def get_model_params(self):
        return self.param_model

    def get_step_params(self):
        return self.param_step

    def sync_param_server(self):
        self.remote_param_server.sync_params.remote(self.agent.behavior_policy_net.state_dict(), self.step)

    def run(self):
        # check if the memory server contains enough data
        while ray.get(self.remote_memory_server.get_size.remote()) < self.start_train_step:
            continue

        # start training
        pbar = tqdm.trange(self.total_time_steps)
        for t in pbar:
            # increase the steps
            self.step = t

            # update the behavior policy
            if not np.mod(self.step, self.update_policy_freq):
                # sample a batch data
                batch_data = ray.get(self.remote_memory_server.sample.remote(self.batch_size))
                self.agent.update_behavior_policy(batch_data)
                self.sync_param_server()

            # update the target policy
            if not np.mod(self.step, self.update_target_freq):
                # update the target network
                self.agent.update_target_policy()

            if not np.mod(self.step, 200):
                G = eval_policy(env_params, self.agent)
                self.returns.append(G)
                self.steps.append(t)

            # print information
            pbar.set_description(
                    f'Step: {self.step} |'
                    f'G: {G:.2f} | '
                    f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
            )
        np.save("./w3_returns.npy", self.returns)


ray.init()  # init the ray

if __name__ == '__main__':
    total_time_steps = 100000

    # init the params
    env_params = {
        'env_name': 'CartPole-v0',
        'max_episode_time_steps': 200,
        'total_time_steps': total_time_steps,
        'act_num': 2,
        'obs_dim': 4
    }

    # init the agent parameters
    agent_params = {
        'agent_id': None,
        'agent_model': None,
        'dqn_mode': 'double',
        'use_obs': False,
        'polyak': 0.95,
        'device': 'cpu',
        'lr': 1e-3,
        'gamma': 0.9995,
        'use_soft_update': False
    }

    # initialize parameters for training
    train_params = {
        'agent': None,
        'memory_size': 50000,
        'batch_size': 256,
        'episode_time_steps': 200,
        'total_time_steps': total_time_steps,
        'lr': 1e-3,
        'update_target_freq': 2000,
        'update_policy_freq': 4,
        'start_train_step': 1000
    }

    # create the remote memory server
    remote_memory_server = MemoryServer.remote(train_params['memory_size'])
    # create the agent
    agent = DQNAgent(env_params, agent_params)
    train_params['agent'] = copy.deepcopy(agent)
    agent_params['agent_model'] = copy.deepcopy(agent)
    # create the remote parameter server
    remote_param_server = ParamServer.remote(agent.behavior_policy_net.state_dict())
    # create the remote learner server
    remote_learner = Learner.remote(train_params, remote_param_server, remote_memory_server)

    # create the actors
    actor_num = 3
    actors = []
    for i in range(actor_num):
        agent_params['agent_id'] = i
        agent_params['agent_model'] = DQNAgent(env_params, agent_params)
        actors.append(Actor.remote(agent_params, env_params, remote_param_server, remote_memory_server))

    processes = [remote_learner]
    for actor in actors:
        processes.append(actor)

    processes_running = [p.run.remote() for p in processes]
    ray.wait(processes_running)