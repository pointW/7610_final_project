import torch
import gym
import numpy as np
from torch import nn
from model.DeepNetworks import DeepQNet
from agent.DQNAgent import DQNAgent
from utils.torch_utils import WeightedHuberLoss


class DQNAgentPER(DQNAgent):
    # initialize the agent
    def __init__(self,
                 env_params=None,
                 agent_params=None,
                 ):
        super().__init__(env_params, agent_params)
        self.weighted_huber_loss = WeightedHuberLoss()

    # update behavior policy
    def update_behavior_policy(self, batch_data):
        batch_data, weights, idxes = batch_data
        weights = torch.from_numpy(weights.copy()).float().unsqueeze(1).to(self.device)

        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the q value estimation using the behavior network
        pred_q_value = self.behavior_policy_net(obs_tensor)
        pred_q_value = pred_q_value.gather(dim=1, index=actions_tensor)

        # compute the TD target using the target network
        if self.dqn_mode == 'vanilla':
            # compute the TD target using vanilla method: TD = r + gamma * max a' Q(s', a')
            # no gradient should be tracked
            with torch.no_grad():
                max_next_q_value = self.target_policy_net(next_obs_tensor).max(dim=1)[0].view(-1, 1)
                td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
        else:
            # compute the TD target using double method: TD = r + gamma * Q(s', argmaxQ_b(s'))
            with torch.no_grad():
                max_next_actions = self.behavior_policy_net(next_obs_tensor).max(dim=1)[1].view(-1, 1).long()
                max_next_q_value = self.target_policy_net(next_obs_tensor).gather(dim=1, index=max_next_actions).view(
                    -1, 1)
                td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
                td_target_value = td_target_value.detach()

        # compute the loss
        td_loss = self.weighted_huber_loss(pred_q_value, td_target_value, weights, torch.ones(obs_tensor.size(0), 1).to(self.device))

        with torch.no_grad():
            td_error = torch.abs(pred_q_value - td_target_value)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss, td_error


