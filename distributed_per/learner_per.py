import ray
import numpy as np
from utils.Schedule import LinearSchedule


class Learner(object):
    def __init__(self, learn_params, env_params, param_server_remote, memory_server_remote):
        # remote servers
        self.remote_memory_server = memory_server_remote
        self.remote_param_server = param_server_remote

        # learning params
        self.use_per = learn_params['use_prioritized_replay']
        self.worker_num = learn_params['worker_num']
        self.epochs = learn_params['epochs']  # training epochs
        self.agent = learn_params['agent']  # model of the agent
        self.batch_size = learn_params['batch_size']  # batch size
        self.start_train_memory_size = learn_params['start_train_memory_size']  # minimal memory size

        # schedule
        self.scheduled_eps = 1
        self.schedule = LinearSchedule(1, 0.01, self.epochs / 2)

        # testing env
        self.env_params = env_params
        self.test_env = env_params['env_fn']()

        # current training step
        self.step = 0

        # save results
        self.returns = []

    def sync_param_server(self):
        self.remote_param_server.sync_learner_model_params.remote(self.agent.behavior_policy_net.state_dict(),
                                                                  self.scheduled_eps, self.step)

    def update_priorities_on_memory_server(self, batch_data, td_err):
        # get the indices
        _, _, _, _, _, _, indices = batch_data
        # compute the priorities
        new_priorities = np.abs(td_err) + 1e-6
        # update the priorities of the sampled batch data
        self.remote_memory_server.update_priorities.remote(indices, new_priorities)

    def eval_policy(self, episode):
        old_eps = self.agent.eps
        returns = []
        self.agent.eps = 0
        for _ in range(episode):
            rewards = []
            obs = self.test_env.reset()
            for t in range(self.env_params['max_episode_time_steps']):
                action = self.agent.get_action(obs)
                next_obs, reward, done, _ = self.test_env.step(action)
                rewards.append(reward)
                if done:
                    break
                else:
                    obs = next_obs

            # compute returns
            G = 0
            for r in reversed(rewards):
                G = r + self.agent.gamma * G
            returns.append(G)

        self.agent.eps = old_eps
        return returns

    def update(self):
        self.step += 1
        batch_data = ray.get(self.remote_memory_server.sample.remote(self.batch_size))
        print(batch_data)
        # compute the epsilon
        self.scheduled_eps = self.schedule.get_value(self.step)
        # update the behavior policy
        if self.use_per:
            loss, td_err = self.agent.update_behavior_policy(batch_data)
            # update the priorities of the batch data on the remote memory server
            self.update_priorities_on_memory_server(batch_data, td_err)
        else:
            loss = self.agent.update_behavior_policy(batch_data)
        # send to the parameter server
        self.sync_param_server()
        return loss

    def update_target(self):
        self.agent.update_target_policy()
