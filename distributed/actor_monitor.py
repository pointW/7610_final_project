from copy import deepcopy
import time
import ray


class ActorMonitor:
    def __init__(self, actor_num, actor_class, agents, agent_params, env_params, remote_param_server,
                 remote_memory_server, remote_actor_state_server, actor_restart_t=10):
        self.actor_num = actor_num
        self.actor_class = actor_class
        self.agents = agents
        self.agent_params = [deepcopy(agent_params) for _ in range(actor_num)]
        self.env_params = env_params
        self.remote_param_server = remote_param_server
        self.remote_memory_server = remote_memory_server
        self.remote_actor_state_server = remote_actor_state_server

        self.actors = []
        self.actor_restart_t = actor_restart_t

        for i in range(actor_num):
            self.agent_params[i]['agent_id'] = i
            self.agent_params[i]['agent_model'] = agents[i]
            self.actors.append(actor_class.remote(self.agent_params[i], self.env_params, self.remote_param_server,
                                                  self.remote_memory_server, self.remote_actor_state_server))

        self.actor_processes = [a.run.remote() for a in self.actors]

    def check_and_restart_actors(self):
        now = time.time()
        actor_alive_time = ray.get(self.remote_actor_state_server.get_actor_alive_time_list.remote())
        for i in range(self.actor_num):
            if (now - actor_alive_time[i]) >= self.actor_restart_t:
                print('Actor {} not reachable, restarting'.format(i))
                ray.kill(self.actors[i])
                self.actors[i] = self.actor_class.remote(self.agent_params[i], self.env_params, self.remote_param_server,
                                                         self.remote_memory_server, self.remote_actor_state_server)
                self.remote_actor_state_server.update_alive.remote(i)
                self.actor_processes[i] = self.actors[i].run.remote()
