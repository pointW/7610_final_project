import torch
import gym
import numpy as np
from torch import nn
from model.DeepNetworks import DeepQNet


# # customized weight initialization
# def customized_weights_init(m):
#     # compute the gain
#     gain = nn.init.calculate_gain('relu')
#     # init the convolutional layer
#     if isinstance(m, nn.Conv2d):
#         # init the params using uniform
#         nn.init.xavier_uniform_(m.weight, gain=gain)
#         nn.init.constant_(m.bias, 0)
#     # init the linear layer
#     if isinstance(m, nn.Linear):
#         # init the params using uniform
#         nn.init.xavier_uniform_(m.weight, gain=gain)
#         nn.init.constant_(m.bias, 0)


class DDPGAgent(object):
    # initialize the agent
    def __init__(self,
                 env_params=None,
                 agent_params=None,
                 ):
        # save the parameters
        self.env_params = env_params
        self.agent_params = agent_params

        # environment parameters
        self.action_space = np.linspace(0, env_params['act_num'], env_params['act_num'], endpoint=False).astype('uint8')
        self.action_dim = env_params['act_num']
        self.obs_dim = env_params['obs_dim']

        # create behavior policy and target networks
        # self.dqn_mode = agent_params['dqn_mode']
        self.use_obs = agent_params['use_obs']
        self.gamma = agent_params['gamma']

        #actor network = behavior netowrk
        self.behavior_policy_net = DeepQNet(self.obs_dim, self.action_dim, output_activation = nn.Tanh())
        self.target_policy_net = DeepQNet(self.obs_dim, self.action_dim, output_activation = nn.Tanh())
        #critic netowrk is seprate, but the actors do not need to have it for execution
        self.critic_net = DeepQNet(self.obs_dim+self.action_dim, self.action_dim)
        self.critic_target_net = DeepQNet(self.obs_dim+self.action_dim, self.action_dim)

        # initialize target networks with actor and critic network parameters
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        # send the agent to a specific device: cpu or gpu
        self.device = torch.device(agent_params['device'])
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)
        self.critic_net.to(self.device)
        self.critic_target_net.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=self.agent_params['lr'])
        self.critic_optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=self.agent_params['lr'])

        # other parameters
        self.eps = 1

        #action randomness
        self.action_std_div = 0.5

    # get action
    def get_action(self, obs):
        randomness = (0.0 if self.eps == 0.0 else self.action_std_div) #random behavior is constant except during test time
        obs = self._arr_to_tensor(obs).view(1, -1)
        with torch.no_grad():
            action = [self.behavior_policy_net(obs).item() + np.random.normal(0.0,randomness)] #random perturbations for off policy learning
        return action

    # update behavior policy
    def update_behavior_policy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # for d in ['obs','action','next_obs','reward','done']:
        #     tens = batch_data_tensor[d]
        #     print(tens.shape)

        # compute the q value estimation using the behavior network
        # pred_q_value = self.behavior_policy_net(obs_tensor)
        with torch.no_grad():
            next_q_values = self.critic_target_net(torch.cat((next_obs_tensor,self.target_policy_net(next_obs_tensor)),1))
            # pred_q_value = pred_q_value.gather(dim=1, index=actions_tensor)
            target_q_values = rewards_tensor + self.agent_params['gamma']*(1-dones_tensor)*next_q_values

        #critic update
        self.critic_optimizer.zero_grad()
        q_batch = self.critic_net(torch.cat((obs_tensor,actions_tensor),1))
        td_loss = torch.nn.functional.mse_loss(q_batch, target_q_values)
        td_loss.backward()
        self.critic_optimizer.step()

        #actor update
        self.optimizer.zero_grad()
        policy_loss = -self.critic_net(torch.cat((obs_tensor,self.behavior_policy_net(obs_tensor)),1)).mean()
        policy_loss.backward()
        self.optimizer.step()

    # update update target policy
    def update_target_policy(self):
        if self.agent_params['use_soft_update']:  # update the target network using polyak average (i.e., soft update)
            # polyak ~ 0.95
            for param, target_param in zip(self.behavior_policy_net.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_(
                    (1 - self.agent_params['polyak']) * param + self.agent_params['polyak'] * target_param)

            for param, target_param in zip(self.critic_net.parameters(), self.critic_target_net.parameters()):
                target_param.data.copy_(
                    (1 - self.agent_params['polyak']) * param + self.agent_params['polyak'] * target_param)

        else:  # hard update
            self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())
            self.critic_target_net.load_state_dict(self.critic_net.state_dict())

    # load trained model
    def load_model(self, model_file, critic_model_file = None):
        # load the trained model
        self.behavior_policy_net.load_state_dict(torch.load(model_file, map_location=self.device))
        self.behavior_policy_net.eval()
        self.critic_net.load_state_dict(torch.load(critic_model_file, map_location=self.device))
        self.critic_net.eval()

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        return batch_data_tensor

    def eval_policy(self, env_params):
        # create environment
        env_test = gym.make(env_params['env_name'])
        # set running statistics
        old_eps = self.eps
        run_num = env_params['run_eval_num']
        returns = []

        self.eps = 0
        for r in range(run_num):
            # reset domain
            obs, rewards = env_test.reset(), []
            # one rollout
            for t in range(env_params['max_episode_time_steps']):
                # get greedy policy
                action = self.get_action(obs)
                # interaction
                next_obs, reward, done, _ = env_test.step(action)
                # save rewards
                rewards.append(reward)
                # check termination
                if done:
                    G = 0
                    for r in reversed(rewards):
                        G = r + 0.9995 * G
                    returns.append(G)
                    break
                else:
                    obs = next_obs

        # reset the epsilon
        self.eps = old_eps
        return np.mean(returns)
