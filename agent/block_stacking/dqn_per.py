from agent.block_stacking.dqn_wrapper import DQNBlock
from utils import torch_utils
import torch
import torch.nn.functional as F


class DQNBlockPER(DQNBlock):
    def __init__(self, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24):
        super().__init__(workspace, heightmap_resolution, device, lr, gamma, sl, num_primitives, patch_size)
        self.weighted_huber_loss = torch_utils.WeightedHuberLoss()

    def update_behavior_policy(self, batch):
        return self.update(batch)

    def _loadBatchToDevice(self, batch):
        batch, weights, idxes = batch
        weights = torch.from_numpy(weights.copy()).float().to(self.device)
        self.loss_calc_dict['weights'] = weights
        self.loss_calc_dict['idxes'] = idxes
        return super()._loadBatchToDevice(batch), weights, idxes
