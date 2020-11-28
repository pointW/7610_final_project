import numpy as np
import tqdm
import gym
from utils.Schedule import LinearSchedule
from utils.ExperienceReplay import ReplayBuffer
from utils.ExperienceReplay import PrioritizedReplayBuffer
from agent.DQNAgent_PER import DQNAgent


def train_dqn_agent(env, agent, trn_params):
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
                f'Buffer: {len(memory)}'
            )

            # reset the environment
            obs, rewards = env.reset(), []
            episode_t = 0
        else:
            obs = next_obs

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
                    batch_data = memory.sample_batch(trn_params['batch_size'], 0)
                    agent.update_behavior_policy(batch_data)

            # update the target network
            if not np.mod(t, trn_params['update_target_freq']):
                agent.update_target_policy()

    # save the results
    if trn_params['use_priority_experience_replay']:
        np.save('./baseline_dqn_per_' + trn_params['env_name'] + '.npy', trn_returns)
    else:
        np.save('./baseline_dqn_' + trn_params['env_name'] + '.npy', trn_returns)


if __name__ == '__main__':

    init_name = 'CartPole-v0'
    init_env = gym.make(init_name)

    # init the params
    env_params = {
        'env_name': init_name,
        'max_episode_time_steps': 200,
        'act_num': init_env.action_space.n,
        'obs_dim': init_env.observation_space.shape[0],
        'run_eval_num': 10
    }

    # init the agent parameters
    agent_params = {
        'agent_id': None,
        'agent_model': None,
        'dqn_mode': 'vanilla',
        'use_obs': False,
        'polyak': 0.95,
        'device': 'cpu',
        'lr': 1e-4,
        'gamma': 0.9995,
        'use_soft_update': False,
        'use_prioritized_replay': False
    }

    # initialize parameters for training
    train_params = {
        'env_name': init_name,
        'agent': None,
        'worker_num': 2,
        'batch_size': 128,
        'memory_size': 50000,
        'total_time_steps': 50000,
        'episode_time_steps': 200,
        'lr': 1e-3,
        'update_target_freq': 2000,
        'update_policy_freq': 4,
        'eval_policy_freq': 100,
        'start_train_step': 1000,
        'use_priority_experience_replay': False
    }

    # create the environment
    my_env = gym.make(env_params['env_name'])
    # create the agent
    my_agent = DQNAgent(env_params, agent_params)

    # train the DQN agent
    train_dqn_agent(my_env, my_agent, train_params)