import ray
import numpy as np
import time

class Actor(object):
    def __init__(self, agent_configs, env_configs, param_server_remote, memory_server_remote, actor_state_server_remote):
        # store the IDs
        self.id = agent_configs['agent_id']

        # store the agent model
        self.agent = agent_configs['agent_model']

        # init remote servers
        self.remote_param_server = param_server_remote
        self.remote_memory_server = memory_server_remote
        self.remote_actor_state_server = actor_state_server_remote

        # local experience replay buffer
        self.local_buffer = []
        self.buffer_size = env_configs['max_episode_time_steps']

        # environment parameters
        self.env = env_configs['env_fn']()
        self.episode_time_steps = env_configs['max_episode_time_steps'] - 1
        self.total_train_steps = ray.get(self.remote_param_server.get_total_train_steps.remote())
        self.current_train_step = 0

        # running indicators
        self.scheduled_eps = 1

        # crash probability
        self.crash_prob = agent_configs['crash_prob']

        # T for reporting alive
        self.report_alive_t = agent_configs['report_alive_t']

        # last alive reporting time
        self.last_alive_report_time = time.time()

    def update_behavior_policy(self):
        # synchronize the behavior policy with the latest parameters on the parameter server
        self.agent.behavior_policy_net.load_state_dict(ray.get(self.remote_param_server.get_latest_model_params.remote()))
        self.agent.behavior_policy_net.eval()
        # synchronize the scheduled epsilon with the latest epsilon on the parameter server
        self.scheduled_eps = ray.get(self.remote_param_server.get_scheduled_eps.remote())
        # get the current train step
        self.current_train_step = ray.get(self.remote_param_server.get_current_train_step.remote())

    def send_data(self):
        # send the data to the memory server
        self.remote_memory_server.add.remote(self.local_buffer)
        # clear the local memory buffer
        self.local_buffer = []

    def send_alive(self):
        self.remote_actor_state_server.update_alive.remote(self.id)
        self.last_alive_report_time = time.time()

    def run(self):
        # synchronize the parameters
        self.update_behavior_policy()
        # initialize the environment
        obs, rewards = self.env.reset(), []
        # start data collection until the training process terminates
        while self.current_train_step < self.total_train_steps:
            # tell remote_actor_state_server i'm alive
            if time.time() - self.last_alive_report_time > self.report_alive_t:
                self.send_alive()
            # crash with self.crash_prob to simulate actor crash
            if np.random.random() < self.crash_prob:
                print('Actor {}: simulating crash'.format(self.id))
                exit()
            # compute the epsilon
            self.agent.eps = self.scheduled_eps
            # get the action
            action = self.agent.get_action(obs)
            # interaction with the environment
            next_obs, reward, done, _ = self.env.step(action)
            # record rewards
            rewards.append(reward.item())
            # add the local buffer
            self.local_buffer.append((obs, action, reward, next_obs, done))
            # check termination
            if done:
                G = 0
                for r in reversed(rewards):
                    G = r + self.agent.gamma * G
                self.remote_actor_state_server.add_return.remote(G)
                # print(f"Actor {self.id}: G = {G}, Eps = {self.scheduled_eps}")
                # reset environment
                obs, rewards = self.env.reset(), []
                # synchronize the behavior policy
                self.update_behavior_policy()
                # send data to remote memory buffer
                self.send_data()
            else:
                obs = next_obs

        print(f"Actor {self.id} terminates.")

