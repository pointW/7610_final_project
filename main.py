import ray
import tqdm
import gym
import numpy as np
from agent.DQNAgent import DQNAgent
from model.Worker import RolloutWorker
from utils.ExperienceReplay import ReplayBuffer
from utils.Schedule import LinearSchedule
import IPython.terminal.debugger as Debug


def eval_policy(env, policy):
    env_test = gym.make(env['env_name'])
    obs, rewards = env_test.reset(), []
    old_eps = policy.eps

    policy.eps = 0
    for t in range(env['max_episode_time_steps']):
        action = policy.get_action(obs)
        next_obs, reward, done, _ = env_test.step(action)
        rewards.append(reward)
        if done:
            G = 0
            for r in reversed(rewards):
                G = r + 0.9995 * G
            break
        else:
            obs = next_obs

    policy.eps = old_eps
    return G


if __name__ == '__main__':
    # init the params
    env_params = {
        'env_name': 'CartPole-v1',
        'max_episode_time_steps': 500,
        'act_num': 2,
        'obs_dim': 4
    }

    # init the agent parameters
    agent_params = {
        'agent': None,
        'dqn_mode': 'double',
        'use_obs': False,
        'polyak': 0.95,
        'device': 'cpu',
        'lr': 1e-3,
        'gamma': 0.9995,
        'use_soft_update': False
    }

    # initialize parameters for training
    train_params = {
        'memory_size': 50000,
        'batch_size': 128,
        'episode_time_steps': 200,
        'iteration_num': 1000,
        'lr': 1e-3
    }

    # compute schedule
    my_schedule = LinearSchedule(1, 0.01, train_params['iteration_num'] / 2)

    # create the replay buffer
    memory = ReplayBuffer(train_params['memory_size'])
    # init the agent
    agent = DQNAgent(env_params, agent_params)
    # create workers
    worker_num = 6
    workers = [RolloutWorker.remote(env_params) for _ in range(worker_num)]

    pbar = tqdm.trange(train_params['iteration_num'])
    for it in pbar:
        # compute the epsilon
        agent.eps = my_schedule.get_value(it)

        # store the model to object store and return a ID as a reference
        model_id = ray.put(agent)
        # put the model to remote workers
        central_episodes_ids = [
            worker.rollout.remote(model_id) for worker in workers
        ]

        # load all the data to replay buffer
        for j in range(worker_num):
            # wait for ready transition
            episode_ready_ids, central_episodes_ids = ray.wait(central_episodes_ids)
            episodes = ray.get(episode_ready_ids)
            for episode in episodes:
                for trans in episode:
                    obs, act, r, next_obs, d = trans
                    memory.add(obs, act, r, next_obs, d)

                G = eval_policy(env_params, agent)

                # print information
                pbar.set_description(
                    f'G: {G:.2f} | '
                    f'Buffer: {len(memory)}'
                )

            batch_data = memory.sample_batch(train_params['batch_size'])
            agent.update_behavior_policy(batch_data)
            if not np.mod(it, 50):
                agent.update_target_policy()
