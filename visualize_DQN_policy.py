from agent.DQNAgent import DQNAgent
import torch
import gym
from gym import wrappers
import argparse


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")

    return parser.parse_args()


if __name__ == '__main__':
    input_args = parse_input()
    env_to_wrap = gym.make(input_args.env)
    env = wrappers.Monitor(env_to_wrap, './' + input_args.env + '.mp4', force=True)
    # init the params
    env_params = {
        'env_name': input_args.env,
        'max_episode_time_steps': 500,
        'act_num': env.action_space.n,
        'obs_dim': env.observation_space.shape[0],
        'run_eval_num': 10
    }

    # init the agent parameters
    agent_params = {
        'agent_id': None,
        'agent_model': None,
        'dqn_mode': 'double',
        'use_obs': False,
        'polyak': 0.95,
        'device': 'cpu',
        'lr': 1e-3,
        'gamma': 0.9995,
        'use_soft_update': False,
        'crash_prob': 0,
        'report_alive_t': 0.1
    }

    # create agent
    agent = DQNAgent(env_params, agent_params)
    agent.behavior_policy_net.load_state_dict(torch.load('./results/models/parallel_' + input_args.env + '_model.pt',
                                                         map_location='cpu'))
    agent.behavior_policy_net.eval()

    obs = env.reset()
    agent.eps = 0
    frames = []
    for t in range(500):
        action = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        env.render('rgb_array')

        if done:
            obs = env.reset()
        else:
            obs = next_obs

    env.close()
    env_to_wrap.close()

