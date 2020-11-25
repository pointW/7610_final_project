import ray
import numpy as np
import random
import time


@ray.remote
class Actor(object):
    def __init__(self, actor_id, remote_param_server, remote_memory):
        # store the IDs
        self.id = actor_id
        self.remote_leaner = remote_param_server
        self.remote_memory = remote_memory
        self.total_time_steps = 10000

        self.local_buffer = []

        # parameters to be updated
        self.param_model = -1
        self.param_step = -1

    def update_from_remote_param_server(self):
        self.param_model = ray.get(self.remote_leaner.get_model_params.remote())
        self.param_step = ray.get(self.remote_leaner.get_step_params.remote())

    def send_to_remote_memory(self):
        self.update_from_remote_param_server()
        while self.param_step < self.total_time_steps:
            self.update_from_remote_param_server()
            print(f"Actor {self.id}: model={self.param_model}, step={self.param_step}")
            max_time_steps = 10
            for i in range(max_time_steps):
                item = random.random()
                self.local_buffer.append(item)
                if len(self.local_buffer) > 64:
                    self.remote_memory.add.remote(self.local_buffer)  # remote call to save
                    self.local_buffer = []

    def run(self):
        self.send_to_remote_memory()


@ray.remote
class Memory(object):
    def __init__(self, memory_size):
        self.size = memory_size
        self.storage = [1]

    def get_size(self):
        return len(self.storage)

    def add(self, item_list):
        [self.storage.append(item) for item in item_list]

    def sample(self):
        return random.sample(self.storage, 1)[0]


@ray.remote
class ParamServer(object):
    def __init__(self):
        self.param_model = 0
        self.param_step = 0

    def update_params(self, param_model, param_step):
        self.param_model = param_model
        self.param_step = param_step

    def get_model_params(self):
        return self.param_model

    def get_step_params(self):
        return self.param_step


@ray.remote
class Learner(object):
    def __init__(self, remote_param_server, remote_memory):
        self.param_model = 0
        self.param_step = 0
        self.total_time_steps = 10000
        self.remote_memory = remote_memory
        self.remote_param_server = remote_param_server

    def get_model_params(self):
        return self.param_model

    def get_step_params(self):
        return self.param_step

    def update_model(self):
        self.param_model = ray.get(self.remote_memory.sample.remote())

    def sync_param_server(self):
        self.remote_param_server.update_params.remote(self.param_model, self.param_step)

    def run(self):
        for i in range(10000):
            time_start = time.time()
            self.update_model()
            self.param_step += 1
            print(f"Leaner: model={self.param_model}, step={self.param_step}, memory_size={ray.get(self.remote_memory.get_size.remote())}")
            self.sync_param_server()


ray.init()  # init the ray

if __name__ == '__main__':
    remote_memory = Memory.remote(2000)
    remote_param_server = ParamServer.remote()
    remote_learner = Learner.remote(remote_param_server, remote_memory)
    actor_num = 2
    actors = [Actor.remote(i, remote_param_server, remote_memory) for i in range(actor_num)]

    processes = [remote_learner]
    for actor in actors:
        processes.append(actor)

    processes_running = [p.run.remote() for p in processes]

    ray.wait(processes_running)