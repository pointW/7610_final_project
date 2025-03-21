import numpy as np
import torch
import torch.nn.functional as F

class BaseAgent:
    def __init__(self, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24):
        self.workspace = workspace
        self.heightmap_resolution = heightmap_resolution
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.sl = sl
        self.num_primitives = num_primitives
        self.patch_size = patch_size

        self.patch_div_factor = 1
        self.patch_mul_factor = 300

        self.heightmap_size = 90
        self.padding = 128

        self.loss_calc_dict = {}

        self.fcn = None
        self.target_fcn = None
        self.fcn_optimizer = None

        self.networks = []
        self.target_networks = []
        self.optimizers = []

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.01):
        raise NotImplementedError

    def getActionFromPlan(self, plan):
        raise NotImplementedError

    def decodeActions(self, *args):
        raise NotImplementedError

    def calcTDLoss(self):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def forwardFCN(self, states, in_hand, obs, target_net=False, to_cpu=False):
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        q1 = self.fcn if not target_net else self.target_fcn
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        q_value_maps, obs_encoding = q1(obs, in_hand)
        q_value_maps = q_value_maps[torch.arange(0, states.size(0)), states.long(), padding_width: -padding_width,
                       padding_width: -padding_width]
        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps, obs_encoding

    def encodeInHand(self, input_img, in_hand_img):
        if input_img.size(2) == in_hand_img.size(2):
            return torch.cat((input_img, in_hand_img), dim=1)
        else:
            resized_in_hand = F.interpolate(in_hand_img, size=(input_img.size(2), input_img.size(3)),
                                            mode='nearest')
        return torch.cat((input_img, resized_in_hand), dim=1)

    def getPatch(self, obs, center_pixel, rz):
        batch_size = obs.size(0)
        img_size = obs.size(2)
        transition = (center_pixel - obs.size(2) / 2).float().flip(1)
        transition_scaled = transition / obs.size(2) * 2

        # affine_mat = torch.eye(2).unsqueeze(0).expand(batch_size, -1, -1).float()
        # if obs.is_cuda:
        #     affine_mat = affine_mat.to(self.device)
        # affine_mat = torch.cat((affine_mat, transition_scaled.unsqueeze(2).float()), dim=2)
        affine_mats = []
        for rot in rz:
            affine_mat = np.asarray([[np.cos(rot), np.sin(rot)],
                                     [-np.sin(rot), np.cos(rot)]])
            affine_mat.shape = (2, 2, 1)
            affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float().to(self.device)
            affine_mats.append(affine_mat)
        affine_mat = torch.cat(affine_mats)
        if obs.is_cuda:
            affine_mat = affine_mat.to(self.device)
        affine_mat = torch.cat((affine_mat, transition_scaled.unsqueeze(2).float()), dim=2)
        flow_grid = F.affine_grid(affine_mat, obs.size())
        transformed = F.grid_sample(obs, flow_grid, mode='bilinear', padding_mode='zeros')
        patch = transformed[:, :,
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2),
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2)]
        return patch

    def normalizePatch(self, patch):
        normalized_patch = patch / self.patch_div_factor
        normalized_patch *= self.patch_mul_factor
        return normalized_patch

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
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs[0])
            in_hands.append(d.obs[1])
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs[0])
            next_in_hands.append(d.next_obs[1])
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        in_hand_tensor = torch.stack(in_hands).to(self.device)
        if len(in_hand_tensor.shape) == 3:
            in_hand_tensor = in_hand_tensor.unsqueeze(1)
        xy_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        next_in_hands_tensor = torch.stack(next_in_hands).to(self.device)
        if len(next_in_hands_tensor.shape) == 3:
            next_in_hands_tensor = next_in_hands_tensor.unsqueeze(1)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = (image_tensor, in_hand_tensor)
        self.loss_calc_dict['action_idx'] = xy_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = (next_obs_tensor, next_in_hands_tensor)
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, (image_tensor, in_hand_tensor), xy_tensor, rewards_tensor, next_states_tensor, \
               (next_obs_tensor, next_in_hands_tensor), non_final_masks, step_lefts_tensor, is_experts_tensor

    def _loadLossCalcDict(self):
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def updateTarget(self):
        for i in range(len(self.networks)):
            self.target_networks[i].load_state_dict(self.networks[i].state_dict())

    def loadModel(self, path_pre):
        for i in range(len(self.networks)):
            path = path_pre + '_q{}.pt'.format(i)
            print('loading {}'.format(path))
            self.networks[i].load_state_dict(torch.load(path))
        self.updateTarget()

    def saveModel(self, path):
        for i in range(len(self.networks)):
            torch.save(self.networks[i].state_dict(), '{}_q{}.pt'.format(path, i))

    def getSaveState(self):
        state = {}
        for i in range(len(self.networks)):
            state['q{}'.format(i)] = self.networks[i].state_dict()
            state['q{}_target'.format(i)] = self.target_networks[i].state_dict()
            state['q{}_optimizer'.format(i)] = self.optimizers[i].state_dict()
        return state

    def loadFromState(self, save_state):
        for i in range(len(self.networks)):
            self.networks[i].load_state_dict(save_state['q{}'.format(i)])
            self.target_networks[i].load_state_dict(save_state['q{}_target'.format(i)])
            self.optimizers[i].load_state_dict(save_state['q{}_optimizer'.format(i)])

    def train(self):
        for i in range(len(self.networks)):
            self.networks[i].train()

    def eval(self):
        for i in range(len(self.networks)):
            self.networks[i].eval()

    def getModelStr(self):
        return str(self.networks)
