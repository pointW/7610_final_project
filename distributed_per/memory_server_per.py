from utils.ExperienceReplay import ReplayBuffer
from utils.ExperienceReplay import PrioritizedReplayBuffer
import numpy as np


class MemoryServer(object):
    def __init__(self, memory_params):
        self.params = memory_params
        if memory_params['type'] == 'per':
            self.storage = PrioritizedReplayBuffer(memory_params['size'], memory_params['alpha'])
        else:
            self.storage = ReplayBuffer(memory_params['size'])  # create a replay buffer

    def get_size(self):
        return len(self.storage)  # get the size of the replay buffer

    def add(self, item_list):  # add transitions to the replay buffer
        for item in item_list:
            obs, act, reward, next_obs, d = item
            self.storage.add(obs, act, reward, next_obs, d)

    def sample(self, batch_size, val):  # sample a batch with size batch_size
        return self.storage.sample_batch(batch_size, val)

    def update_priorities(self, batch_indices, new_priorities):
        # update the priorities
        self.storage.update_priorities(batch_indices, new_priorities)
