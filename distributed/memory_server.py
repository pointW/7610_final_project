from utils.ExperienceReplay import ReplayBuffer


class MemoryServer(object):
    def __init__(self, size):
        self.size = size  # size of the memory
        self.storage = ReplayBuffer(size)  # create a replay buffer

    def get_size(self):
        return len(self.storage)  # get the size of the replay buffer

    def add(self, item_list):  # add transitions to the replay buffer
        for item in item_list:
            obs, act, reward, next_obs, d = item
            self.storage.add(obs, act, reward, next_obs, d)

    def sample(self, batch_size):  # sample a batch with size batch_size
        return self.storage.sample_batch(batch_size)
