from utils.buffer import PrioritizedQLearningBuffer
from utils.buffer import QLearningBuffer


class BlockStackingMemoryServerPER:
    def __init__(self, size, per_alpha):
        self.size = size
        self.storage = PrioritizedQLearningBuffer(size, per_alpha, QLearningBuffer)

    def get_size(self):
        return len(self.storage)

    def add(self, item_list):
        [self.storage.add(t) for t in item_list]

    def sample(self, batch_size, per_beta):
        return self.storage.sample(batch_size, per_beta)

    def update_priorities(self, batch_idxes, new_priorities):
        self.storage.update_priorities(batch_idxes, new_priorities)
