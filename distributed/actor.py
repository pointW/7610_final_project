import ray


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
        self.env = env_configs['env_fn']()
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