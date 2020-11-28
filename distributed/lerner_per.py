import numpy as np
import ray
from distributed.learner import Learner
from utils.Schedule import LinearSchedule
import time


class LearnerPER(Learner):
    def __init__(self, learn_params, env_params, param_server_remote, memory_server_remote):
        super().__init__(learn_params, env_params, param_server_remote, memory_server_remote)
        self.per_init_beta = learn_params['per_beta']
        self.per_beta_schedule = LinearSchedule(self.per_init_beta, 1, self.epochs)

    def update(self):
        self.step += 1
        per_beta = self.per_beta_schedule.get_value(self.step)
        batch_data, weights, batch_idxes = ray.get(self.remote_memory_server.sample.remote(self.batch_size, per_beta))
        # update the behavior policy
        loss, td_error = self.agent.update_behavior_policy((batch_data, weights, batch_idxes))
        new_priorities = np.abs(td_error.cpu()) + 1e-6
        self.remote_memory_server.update_priorities.remote(batch_idxes, new_priorities)

        # compute the epsilon
        self.scheduled_eps = self.schedule.get_value(self.step)
        # send to the parameter server
        self.sync_param_server()
        return loss