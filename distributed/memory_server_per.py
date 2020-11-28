from utils.ExperienceReplay import ReplayBuffer
from utils.buffer import PrioritizedQLearningBuffer
from utils.ExperienceReplay import ReplayBuffer


class MemoryServerPER(object):
    def __init__(self, size, per_alpha):
        self.size = size  # size of the memory
        self.storage = PrioritizedQLearningBuffer(size, per_alpha, ReplayBuffer)

    def get_size(self):
        return len(self.storage)  # get the size of the replay buffer

    def add(self, item_list):  # add transitions to the replay buffer
        for item in item_list:
            obs, act, reward, next_obs, d = item
            self.storage.add(obs, act, reward, next_obs, d)

    def sample(self, batch_size, per_beta):  # sample a batch with size batch_size
        return self.storage.sample(batch_size, per_beta)

    def update_priorities(self, batch_idxes, new_priorities):
        self.storage.update_priorities(batch_idxes, new_priorities)
