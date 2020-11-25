import tqdm
import torch
import numpy as np
from utils.ExperienceReplay import ReplayBuffer
from utils.Schedule import LinearSchedule


class ExperimentGym(object):
    def __init__(self, env_params, agent_params, trn_params):
        # initialize the parameters
        self.env_params = env_params
        self.agent_params = agent_params
        self.trn_params = trn_params

        # initialize the environment
        self.env_trn = self.env_params['env_trn']
        self.env_tst = self.env_params['env_tst']

        # initialize the agent
        self.agent = self.agent_params['agent']

        # initialize the memory
        self.memory = ReplayBuffer(self.trn_params['total_time_steps'])

        # initialize the schedule
        self.schedule = LinearSchedule(1, 0.01, self.trn_params['total_time_steps'] / 3)

        # training statistics
        self.returns = []
        self.steps = []

    def train_agent(self):
        # reset the environment
        obs, rewards = self.env_trn.reset(), []

        episode_t = 0
        pbar = tqdm.trange(self.trn_params['total_time_steps'])
        for t in pbar:
            # get one action
            eps = self.schedule.get_value(t)
            action = self.agent.get_action(obs, eps)

            # interact with the environment
            next_obs, reward, done, _ = self.env_trn.step(action)

            # add to memory
            self.memory.add(obs, action, reward, next_obs, done)

            # store the data
            self.steps.append(t)
            rewards.append(reward)
            episode_t += 1
            obs = next_obs

            # check termination
            if done or episode_t % self.trn_params['episode_time_steps'] == 0:
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G += self.agent_params['gamma'] * r
                self.returns.append(G)
                episode_idx = len(self.returns)

                # print information
                pbar.set_description(
                    f'Episode: {episode_idx} | '
                    f'Steps: {episode_t} |'
                    f'Return: {G:2f} | '
                    f'Eps: {eps} | '
                    f'Buffer: {len(self.memory)}'
                )

                # reset the environment
                obs, rewards, episode_t = self.env_trn.reset(), [], 0

            # train the agent
            if t > self.trn_params['start_train_step']:
                # sample a mini-batch
                batch_data = self.memory.sample_batch(self.trn_params['batch_size'])
                # update the behavior policy
                if not np.mod(t, self.trn_params['update_policy_freq']):
                    self.agent.update_behavior_policy(batch_data)
                # update the target policy
                if not np.mod(t, self.trn_params['update_target_freq']):
                    self.agent.update_target_policy()

        self.save_results()

    def save_results(self):
        np.save('./results/v0_returns.npy', self.returns)
        np.save('./results/v0_steps.npy', self.steps)
        torch.save(self.agent.behavior_policy_net.state_dict(), './results/v0_model.pt')
