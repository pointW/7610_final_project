import numpy as np
import numpy.random as npr
from random import sample
import torch
from copy import deepcopy
import random
from .segment_tree import SumSegmentTree, MinSegmentTree
from .ExperienceReplay import ReplayBuffer


class QLearningBuffer:
    def __init__(self, size):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        batch_indexes = npr.choice(self.__len__(), batch_size).tolist()
        batch = [self._storage[idx] for idx in batch_indexes]
        return batch

    def getSaveState(self):
        return {
            'storage': self._storage,
            'max_size': self._max_size,
            'next_idx': self._next_idx
        }

    def loadFromState(self, save_state):
        self._storage = save_state['storage']
        self._max_size = save_state['max_size']
        self._next_idx = save_state['next_idx']


class PrioritizedQLearningBuffer:
    def __init__(self, size, alpha, buffer_class):
        self.buffer = buffer_class(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def add(self, *args, **kwargs):
        '''
        See ReplayBuffer.store_effect
        '''
        idx = self.buffer._next_idx
        self.buffer.add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self.buffer) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        '''
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        Args:
          - batch_size: How many transitions to sample.
          - beta: To what degree to use importance weights
                  (0 - no corrections, 1 - full correction)

        Returns (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weights, idxes)
          - obs_batch: batch of observations
          - act_batch: batch of actions executed given obs_batch
          - rew_batch: rewards received as results of executing act_batch
          - next_obs_batch: next set of observations seen after executing act_batch
          - done_mask: done_mask[i] = 1 if executing act_batch[i] resulted in
                       the end of an episode and 0 otherwise.
          - weights: Array of shape (batch_size,) and dtype np.float32
                     denoting importance weight of each sampled transition
          - idxes: Array of shape (batch_size,) and dtype np.int32
                   idexes in buffer of sampled experiences
        '''
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        if type(self.buffer) is ReplayBuffer:
            batch = self.buffer._encode_sample(idxes)
        else:
            batch = [self.buffer._storage[idx] for idx in idxes]
        return batch, weights, idxes

    def update_priorities(self, idxes, priorities):
        '''
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Args:
          - idxes: List of idxes of sampled transitions
          - priorities: List of updated priorities corresponding to
                        transitions at the sampled idxes denoted by
                        variable `idxes`.
        '''
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):

            if priority <= 0:
                print("Invalid priority:", priority)
                print("All priorities:", priorities)

            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def getSaveState(self):
        state = self.buffer.getSaveState()
        state.update(
            {
                'alpha': self._alpha,
                'it_sum': self._it_sum,
                'it_min': self._it_min,
                'max_priority': self._max_priority
            }
        )
        return state

    def loadFromState(self, save_state):
        self.buffer.loadFromState(save_state)
        self._alpha = save_state['alpha']
        self._it_sum = save_state['it_sum']
        self._it_min = save_state['it_min']
        self._max_priority = save_state['max_priority']
