import torch
from helping_hands_rl_envs import env_factory


class BlockStackingEnv:
    def __init__(self, config, num_envs=1, render=False):
        config['render'] = render
        self.env = env_factory.createEnvs(num_envs, 'rl', config['simulator'], config['env'], config, {'random_orientation':False})
        self.heightmap_resolution = config['heightmap_resolution']
        self.workspace = config['workspace']

    def step(self, action):
        x = (action[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(action.size(0), 1)
        y = (action[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(action.size(0), 1)
        action = torch.cat((x, y, action[:, 2:3]), dim=1)
        state_, in_hand_, obs_, reward, done = self.env.step(action, auto_reset=False)
        in_hand_ = in_hand_.permute(0, 3, 1, 2)
        obs_ = obs_.permute(0, 3, 1, 2)
        return (state_, in_hand_, obs_), reward, done, None

    def reset(self):
        state, in_hand, obs = self.env.reset()
        obs = obs.permute(0, 3, 1, 2)
        in_hand = in_hand.permute(0, 3, 1, 2)
        return state, in_hand, obs

    def close(self):
        self.env.close()