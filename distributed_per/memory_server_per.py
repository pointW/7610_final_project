from utils.ExperienceReplay import ReplayBuffer
from utils.ExperienceReplay import PrioritizedReplayBuffer


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
        if self.params['type'] == 'per':
            for item in item_list:
                obs, act, reward, next_obs, d = item
                self.storage.add(obs, act, reward, next_obs, d)
        else:
            for item in item_list:
                obs, act, reward, next_obs, d = item
                self.storage.add(obs, act, reward, next_obs, d)

    def sample(self, batch_size):  # sample a batch with size batch_size
        if self.params['type'] == 'per':
            return self.storage.sample_batch(batch_size, 0)
        else:
            return self.storage.sample_batch(batch_size)

    def update_priorities(self, batch_indices, new_priorities):
        self.storage.update_priorities(batch_indices, new_priorities)
