import numpy as np
import tqdm
import gym
import time
import argparse
from utils.Schedule import LinearSchedule
from utils.ExperienceReplay import ReplayBuffer
from utils.ExperienceReplay import PrioritizedReplayBuffer
from agent.DQNAgent_PER import DQNAgent


def eval_policy(env, agent, episode, episode_len):
    old_eps = agent.eps
    returns = []
    agent.eps = 0
    for _ in range(episode):
        rewards = []
        obs = env.reset()
        for t in range(episode_len):
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            else:
                obs = next_obs

        G = 0
        for r in reversed(rewards):
            G = r + agent.gamma * G
        returns.append(G)

        agent.eps = old_eps
        return returns


def train_dqn_agent(env, test_env, agent, trn_params):
    # create the schedule
    schedule = LinearSchedule(0.3, 0.01, trn_params['total_time_steps'] / 2)
    beta_schedule = LinearSchedule(0.4, 1.0, train_params['total_time_steps'])

    # create memory buffer
    if trn_params['use_priority_experience_replay']:
        memory = PrioritizedReplayBuffer(trn_params['memory_size'], 0.6)
    else:
        memory = ReplayBuffer(trn_params['memory_size'])

    # save the training statistics
    trn_returns = []
    episode_t = 0
    # for evaluation
    time_0 = time.time()
    testing_freq = 10
    test_num = 10
    test_returns = []
    # reset the environment
    obs, rewards = env.reset(), []
    pbar = tqdm.trange(trn_params['total_time_steps'])
    for t in pbar:
        # compute the epsilon
        agent.eps = schedule.get_value(t)
        # compute the action
        action = agent.get_action(obs)
        # step in the environment
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward)

        # add transition
        memory.add(obs, action, reward, next_obs, done)

        # check termination
        if done or (episode_t == trn_params['episode_time_steps']):
            # compute the return
            G = 0
            for r in reversed(rewards):
                G = r + agent.gamma * G
            trn_returns.append(G)

            # print training info
            pbar.set_description(
                f'Episode: {len(trn_returns)} | '
                f'Returns: {G} | '
                f'Buffer: {len(memory)} | '
                f'Eval return: {np.mean(test_returns[-5:]) if test_returns else 0:.2f} | '
            )

            # reset the environment
            obs, rewards = env.reset(), []
            episode_t = 0
        else:
            obs = next_obs

        # evaluate the policy
        if time.time() > (len(test_returns) + 1) * testing_freq:
            G = eval_policy(test_env, agent, test_num, trn_params['episode_time_steps'])
            test_returns.append(G)

        # train the agent
        if t > trn_params['start_train_step']:
            # update the policy network
            if not np.mod(t, trn_params['update_policy_freq']):
                if train_params['use_priority_experience_replay']:
                    # sample the batch data
                    batch_data = memory.sample_batch(train_params['batch_size'], beta_schedule.get_value(t))
                    # get the indices
                    _, _, _, _, _, _, indices = batch_data
                    # get the td error
                    _, td_err = agent.update_behavior_policy(batch_data)
                    # update the priorities
                    new_priorities = np.abs(td_err) + 1e-6
                    memory.update_priorities(indices, new_priorities)
                else:
                    batch_data = memory.sample_batch(trn_params['batch_size'])
                    agent.update_behavior_policy(batch_data)

            # update the target network
            if not np.mod(t, trn_params['update_target_freq']):
                agent.update_target_policy()

    # save the results
    if trn_params['use_priority_experience_replay']:
        np.save('./baseline_dqn_per_' + trn_params['env_name'] + '.npy', trn_returns)
    else:
        np.save('./results/returns/baseline_dqn_' + trn_params['env_name'] + '_' + trn_params['dqn_mode'] + '_.npy', trn_returns)

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--dqn_mode", type=str, default="vanilla")
    parser.add_argument("--train_epochs", type=int, default=10000)
    parser.add_argument("--episode_len", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9995)
    parser.add_argument("--memory_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':

    input_args = parse_input()

    init_name = input_args.env
    init_env = gym.make(init_name)

    # init the params
    env_params = {
        'env_name': init_name,
        'max_episode_time_steps': input_args.episode_len,
        'act_num': init_env.action_space.n,
        'obs_dim': init_env.observation_space.shape[0],
        'run_eval_num': 10
    }

    # init the agent parameters
    agent_params = {
        'agent_id': None,
        'agent_model': None,
        'dqn_mode': input_args.dqn_mode,
        'use_obs': False,
        'polyak': 0.95,
        'device': 'cpu',
        'lr': input_args.lr,
        'gamma': input_args.gamma,
        'use_soft_update': False,
        'use_prioritized_replay': False
    }

    # initialize parameters for training
    train_params = {
        'env_name': init_name,
        'dqn_mode': input_args.dqn_mode,
        'agent': None,
        'worker_num': 2,
        'batch_size': input_args.batch_size,
        'memory_size': input_args.memory_size,
        'total_time_steps': input_args.train_epochs,
        'episode_time_steps': input_args.episode_len,
        'lr': input_args.lr,
        'update_target_freq': 2000,
        'update_policy_freq': 4,
        'eval_policy_freq': 100,
        'start_train_step': 1000,
        'use_priority_experience_replay': False
    }

    # create the environment
    my_env = gym.make(env_params['env_name'])
    eval_env = gym.make(env_params['env_name'])
    # create the agent
    my_agent = DQNAgent(env_params, agent_params)

    # train the DQN agent
    train_dqn_agent(my_env, eval_env, my_agent, train_params)
