import ray
import numpy as np
import gym
import tqdm
import copy

from agent.DQNAgent import DQNAgent
from utils.ExperienceReplay import ReplayBuffer
from utils.Schedule import LinearSchedule


@ray.remote
class Actor(object):
    def __init__(self, agent_configs, env_configs, param_server_remote, memory_server_remote):
        # store the IDs
        self.id = agent_configs['agent_id']

        # store the agent model
        self.agent = agent_configs['agent_model']

        # init remote servers
        self.remote_param_server = param_server_remote
        self.remote_memory_server = memory_server_remote

        # local experience replay buffer
        self.local_buffer = []
        self.buffer_size = env_configs['max_episode_time_steps']

        # environment parameters
        self.env = gym.make(env_configs['env_name'])
        self.episode_time_steps = env_configs['max_episode_time_steps']

        # running indicators
        self.scheduled_eps = 1

    def update_behavior_policy(self):
        # synchronize the behavior policy with the latest parameters on the parameter server
        self.agent.behavior_policy_net.load_state_dict(ray.get(self.remote_param_server.get_latest_model_params.remote()))
        self.agent.behavior_policy_net.eval()
        # synchronize the scheduled epsilon with the latest epsilon on the parameter server
        self.scheduled_eps = ray.get(self.remote_param_server.get_scheduled_eps.remote())

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
        self.size = size  # size of the memory
        self.storage = ReplayBuffer(size)  # create a replay buffer

    def get_size(self):
        return len(self.storage)  # get the size of the replay buffer

    def add(self, item_list):  # add transitions to the replay buffer
        for item in item_list:
            obs, act, reward, next_obs, d = item
            self.storage.add(obs, act, reward, next_obs, d)

    def sample(self, batch_size):  # sample a batch with size batch_size
        return self.storage.sample_batch(batch_size)


@ray.remote
class ParamServer(object):
    def __init__(self, model_state_dict):
        self.model_state_dict = model_state_dict  # parameters of the model
        self.eps = 1  # epsilon for exploration

    def sync_learner_model_params(self, model_stat_dict, eps):  # synchronize the parameters with the learner
        self.model_state_dict = model_stat_dict
        self.eps = eps

    def get_latest_model_params(self):  # return the latest model parameters
        return self.model_state_dict

    def get_scheduled_eps(self):  # return the latest epsilon
        return self.eps


@ray.remote
class Learner(object):
    def __init__(self, learn_params, param_server_remote, memory_server_remote):
        # remote servers
        self.remote_memory_server = memory_server_remote
        self.remote_param_server = param_server_remote

        # learning params
        self.worker_num = learn_params['worker_num']
        self.epochs = learn_params['epochs']  # training epochs
        self.agent = learn_params['agent']  # model of the agent
        self.batch_size = learn_params['batch_size']  # batch size
        self.start_train_memory_size = learn_params['start_train_memory_size']  # minimal memory size
        self.update_target_freq = learn_params['update_target_freq']  # target network update frequency
        self.update_policy_freq = learn_params['update_policy_freq']  # policy network update frequency
        self.eval_policy_freq = learn_params['eval_policy_freq']  # evaluate policy frequency

        # schedule
        self.scheduled_eps = 1
        self.schedule = LinearSchedule(1, 0.01, self.epochs / 2)

        # save results
        self.returns = []

    def sync_param_server(self):
        self.remote_param_server.sync_learner_model_params.remote(self.agent.behavior_policy_net.state_dict(),
                                                                  self.scheduled_eps)

    def run(self):
        # check if the memory server contains enough data
        while ray.get(self.remote_memory_server.get_size.remote()) < self.start_train_memory_size:
            continue

        # start training
        pbar = tqdm.trange(self.epochs)
        for ep in pbar:
            # sample a batch data
            batch_data = ray.get(self.remote_memory_server.sample.remote(self.batch_size))

            # update policy network
            if not np.mod(ep, self.update_policy_freq):
                # compute the epsilon
                self.scheduled_eps = self.schedule.get_value(ep)
                # update the behavior policy
                self.agent.update_behavior_policy(batch_data)
                # send to the parameter server
                self.sync_param_server()

            # update the target policy
            if not np.mod(ep, self.update_target_freq):
                # update the target network
                self.agent.update_target_policy()

            if not np.mod(ep, self.eval_policy_freq):
                G = self.agent.eval_policy(env_params)
                self.returns.append(G)
                # print information
                pbar.set_description(
                    f'Epoch: {ep} |'
                    f'G: {G:.2f} | '
                    f'Eps: {self.scheduled_eps:.3f} | '
                    f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
                )

        np.save(f'./w{self.worker_num}_returns.npy', self.returns)


ray.init()  # init the ray

if __name__ == '__main__':
    total_time_steps = 100000

    # init the params
    env_params = {
        'env_name': 'CartPole-v0',
        'max_episode_time_steps': 200,
        'act_num': 2,
        'obs_dim': 4,
        'run_eval_num': 10
    }

    # init the agent parameters
    agent_params = {
        'agent_id': None,
        'agent_model': None,
        'dqn_mode': 'vanilla',
        'use_obs': False,
        'polyak': 0.95,
        'device': 'cpu',
        'lr': 1e-4,
        'gamma': 0.9995,
        'use_soft_update': False
    }

    # initialize parameters for training
    train_params = {
        'agent': None,
        'worker_num': 1,
        'memory_size': 50000,
        'batch_size': 128,
        'epochs': 50000,
        'lr': 1e-3,
        'update_target_freq': 2000,
        'update_policy_freq': 1,
        'eval_policy_freq': 100,
        'start_train_memory_size': 1000
    }

    # create the remote memory server
    remote_memory_server = MemoryServer.remote(train_params['memory_size'])

    # create the agent
    agent = DQNAgent(env_params, agent_params)
    agent_params['agent_model'] = copy.deepcopy(agent)

    # create the remote parameter server
    remote_param_server = ParamServer.remote(agent.behavior_policy_net.state_dict())

    # create the remote learner server
    train_params['agent'] = copy.deepcopy(agent)
    remote_learner = Learner.remote(train_params, remote_param_server, remote_memory_server)

    # create the actors
    actor_num = train_params['worker_num']
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