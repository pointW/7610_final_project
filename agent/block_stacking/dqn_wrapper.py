import torch
from agent.block_stacking.dqn_2d import DQN2D
from network.models import ResUCatShared


class DQNBlock(DQN2D):
    def __init__(self, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24):
        super().__init__(workspace, heightmap_resolution, device, lr, gamma, sl, num_primitives, patch_size)
        fcn = ResUCatShared(1, 2, domain_shape=(1, 128, 128), patch_shape=(1, 24, 24)).to(device)
        self.initNetwork(fcn)
        self.eps = 1
        self.behavior_policy_net = self.fcn

    def get_action(self, obs):
        state, in_hand, obs = obs
        q_value_maps, action_idx, actions = self.getEGreedyActions(state, in_hand, obs, self.eps)
        action_idx = torch.cat((action_idx, state.unsqueeze(1)), dim=1).long()
        return action_idx

    def update_behavior_policy(self, batch):
        return self.update(batch)

    def update_target_policy(self):
        self.updateTarget()

    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        in_hands = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        next_in_hands = []
        dones = []
        for d in batch:
            states.append(d[0][0])
            images.append(d[0][2])
            in_hands.append(d[0][1])
            xys.append(d[1])
            rewards.append(d[2].squeeze())
            next_states.append(d[3][0])
            next_obs.append(d[3][2])
            next_in_hands.append(d[3][1])
            dones.append(d[4])
        states_tensor = torch.cat(states).long().to(self.device)
        image_tensor = torch.cat(images).to(self.device)
        in_hand_tensor = torch.cat(in_hands).to(self.device)
        xy_tensor = torch.cat(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.cat(next_states).long().to(self.device)
        next_obs_tensor = torch.cat(next_obs).to(self.device)
        next_in_hands_tensor = torch.cat(next_in_hands).to(self.device)
        dones_tensor = torch.cat(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = (image_tensor, in_hand_tensor)
        self.loss_calc_dict['action_idx'] = xy_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = (next_obs_tensor, next_in_hands_tensor)
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = None
        self.loss_calc_dict['is_experts'] = None

        return states_tensor, (image_tensor, in_hand_tensor), xy_tensor, rewards_tensor, next_states_tensor, \
               (next_obs_tensor, next_in_hands_tensor), non_final_masks, None, None
