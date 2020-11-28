from agent.block_stacking.dqn_wrapper import DQNBlock
from utils import torch_utils
import torch
import torch.nn.functional as F


class DQNBlockPER(DQNBlock):
    def __init__(self, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24):
        super().__init__(workspace, heightmap_resolution, device, lr, gamma, sl, num_primitives, patch_size)
        self.weighted_huber_loss = torch_utils.WeightedHuberLoss()

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        weights = self.loss_calc_dict['weights']
        pixel = action_idx[:, 0:2]

        with torch.no_grad():
            q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs[1], next_obs[0], target_net=True)
            x_star = torch_utils.argmax2d(q1_map_prime).long()
            q_prime = q1_map_prime[torch.arange(0, batch_size), x_star[:, 0], x_star[:, 1]]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q1_output, obs_encoding = self.forwardFCN(states, obs[1], obs[0])
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]

        self.loss_calc_dict['q1_output'] = q1_output

        # q1_td_loss = F.smooth_l1_loss(q1_pred, q_target)
        q1_td_loss = self.weighted_huber_loss(q1_pred, q_target, weights, torch.ones(batch_size).to(self.device))
        td_loss = q1_td_loss

        with torch.no_grad():
            td_error = torch.abs(q1_pred - q_target)

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error

    def update_behavior_policy(self, batch):
        return self.update(batch)

    def _loadBatchToDevice(self, batch):
        batch, weights, idxes = batch
        weights = torch.from_numpy(weights.copy()).float().to(self.device)
        self.loss_calc_dict['weights'] = weights
        self.loss_calc_dict['idxes'] = idxes
        return super()._loadBatchToDevice(batch), weights, idxes
