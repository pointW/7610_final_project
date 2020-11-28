import ray
import numpy as np
import gym
import tqdm
import copy
import time

from agent.DQNAgent import DQNAgent
from agent.DQNAgent_per import DQNAgentPER

from distributed.actor import Actor
from distributed.learner import Learner
from distributed.lerner_per import LearnerPER
from distributed.memory_server import MemoryServer
from distributed.memory_server_per import MemoryServerPER
from distributed.param_server import ParamServer
from distributed.actor_state_server import ActorStateServer
from distributed.actor_monitor import ActorMonitor

if __name__ == '__main__':
    ray.init()  # init the ray

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
        'crash_prob': 0,
        'report_alive_t': 0.1
    }

    # initialize parameters for training
    train_params = {
        'agent': None,
        'worker_num': 2,
        'memory_size': 50000,
        'batch_size': 128,
        'epochs': 10000,
        'lr': 1e-3,
        'update_target_freq': 2000,
        'update_policy_freq': 1,
        'eval_policy_freq': 100,
        'start_train_memory_size': 1000,
        'per_alpha': 0.6,
        'per_beta': 0.4
    }

    PER = True

    def env_fn():
        return gym.make(env_params['env_name'])
    env_params['env_fn'] = env_fn

    # create the remote memory server
    if PER:
        remote_memory_server = ray.remote(num_cpus=1)(MemoryServerPER).remote(train_params['memory_size'], train_params['per_alpha'])
    else:
        remote_memory_server = ray.remote(num_cpus=1)(MemoryServer).remote(train_params['memory_size'])

    # create the agent
    if PER:
        agent = DQNAgentPER(env_params, agent_params)
    else:
        agent = DQNAgent(env_params, agent_params)
    agent_params['agent_model'] = copy.deepcopy(agent)

    # create the remote parameter server
    remote_param_server = ray.remote(num_cpus=1)(ParamServer).remote(agent.behavior_policy_net.state_dict(), train_params['epochs'])

    # create the remote learner server
    train_params['agent'] = copy.deepcopy(agent)
    if PER:
        remote_learner = ray.remote(num_cpus=1)(LearnerPER).remote(train_params, env_params, remote_param_server,
                                                                   remote_memory_server)
    else:
        remote_learner = ray.remote(num_cpus=1)(Learner).remote(train_params, env_params, remote_param_server, remote_memory_server)

    remote_actor_state_server = ray.remote(num_cpus=1)(ActorStateServer).remote(train_params['worker_num'])
    # create the actors
    actor_agents = []
    for i in range(train_params['worker_num']):
        actor_agents.append(DQNAgent(env_params, agent_params))
    actor_monitor = ActorMonitor(train_params['worker_num'], ray.remote(num_cpus=1)(Actor), actor_agents, agent_params, env_params,
                                 remote_param_server, remote_memory_server, remote_actor_state_server, 2)

    test_returns = []
    while ray.get(remote_memory_server.get_size.remote()) < train_params['start_train_memory_size']:
        actor_monitor.check_and_restart_actors()
        continue

    # start training
    pbar = tqdm.trange(train_params['epochs'])
    G = 0
    t0 = time.time()
    # testing_freq = 2*60
    testing_freq = 10
    for t in range(train_params['epochs']):
        actor_monitor.check_and_restart_actors()
        # sample a batch data
        ray.get(remote_learner.update.remote())
        ray.get(remote_learner.sync_param_server.remote())

        # update the target policy
        if not np.mod(t, train_params['update_target_freq']):
            # update the target network
            ray.get(remote_learner.update_target.remote())

        # if time.time() - t0 > (len(test_returns) + 1) * testing_freq:
        if t % 200 == 0:
            G = np.average(ray.get(remote_learner.eval_policy.remote(10)))
            test_returns.append(G)

        # print information
        pbar.set_description(
            f'Step: {t} |'
            f'G: {np.mean(test_returns[-5:]) if test_returns else 0:.2f} | '
            # f'Buffer: {ray.get(self.remote_memory_server.get_size.remote())}'
        )
        pbar.update()
    np.save("./{}_parallel_returns.npy".format('per' if PER else 'normal'), test_returns)
    ray.wait(actor_monitor.actor_processes)
