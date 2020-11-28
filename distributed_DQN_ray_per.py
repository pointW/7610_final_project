import ray
import numpy as np
import gym
import tqdm
import copy
import time

from agent.DQNAgent_PER import DQNAgent

from distributed_per.actor_per import Actor
from distributed_per.learner_per import Learner
from distributed_per.memory_server_per import MemoryServer
from distributed_per.param_server_per import ParamServer

if __name__ == '__main__':
    ray.init()  # init the ray

    total_time_steps = 100000

    # init the params
    env_params = {
        'env_name': 'CartPole-v0',
        'max_episode_time_steps': 200,
        'act_num': 2,
        'obs_dim': 4,
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
        'use_prioritized_replay': True
    }

    # initialize parameters for training
    train_params = {
        'agent': None,
        'worker_num': 2,
        'batch_size': 4,
        'epochs': 50000,
        'lr': 1e-3,
        'update_target_freq': 2000,
        'update_policy_freq': 1,
        'eval_policy_freq': 100,
        'start_train_memory_size': 1000,
        'use_prioritized_replay': True
    }

    memory_params = {
        'type': 'vanilla',
        'size': 50000,
        'alpha': 1  # {alpha in [0, 1], 0: no prioritized; 1: full prioritized}
    }

    def env_fn():
        return gym.make(env_params['env_name'])
    env_params['env_fn'] = env_fn

    # create the remote memory server
    remote_memory_server = ray.remote(num_cpus=1)(MemoryServer).remote(memory_params)

    # create the agent
    agent = DQNAgent(env_params, agent_params)
    agent_params['agent_model'] = copy.deepcopy(agent)

    # create the remote parameter server
    remote_param_server = ray.remote(num_cpus=1)(ParamServer).remote(agent.behavior_policy_net.state_dict(), train_params['epochs'])

    # create the remote learner server
    train_params['agent'] = copy.deepcopy(agent)
    remote_learner = ray.remote(num_cpus=1)(Learner).remote(train_params, env_params, remote_param_server, remote_memory_server)

    # create the actors
    actor_num = train_params['worker_num']
    actors = []
    for i in range(actor_num):
        agent_params['agent_id'] = i
        agent_params['agent_model'] = DQNAgent(env_params, agent_params)
        actors.append(ray.remote(num_cpus=1)(Actor).remote(agent_params, env_params, remote_param_server, remote_memory_server))

    processes = []
    for actor in actors:
        processes.append(actor)

    processes_running = [p.run.remote() for p in processes]
    test_returns = []
    while ray.get(remote_memory_server.get_size.remote()) < train_params['start_train_memory_size']:
        continue

    # start training
    pbar = tqdm.trange(train_params['epochs'])
    G = 0
    t0 = time.time()
    # testing_freq = 2*60
    testing_freq = 10
    for t in range(train_params['epochs']):
        # sample a batch data
        ray.get(remote_learner.update.remote())
        ray.get(remote_learner.sync_param_server.remote())

        # update the target policy
        if not np.mod(t, train_params['update_target_freq']):
            # update the target network
            ray.get(remote_learner.update_target.remote())

        if time.time() - t0 > (len(test_returns) + 1) * testing_freq:
            G = np.average(ray.get(remote_learner.eval_policy.remote(10)))
            test_returns.append(G)

        # print information
        pbar.set_description(
            f'Step: {t} |'
            f'G: {np.mean(test_returns[-5:]) if test_returns else 0:.2f} | '
            # f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
        )
        pbar.update()
    np.save("./parallel_per_returns.npy", test_returns)
    ray.wait(processes_running)