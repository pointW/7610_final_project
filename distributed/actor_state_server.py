import time
import numpy as np


class ActorStateServer:
    def __init__(self, actor_num):
        self.actor_alive_time = [time.time() for _ in range(actor_num)]
        self.actor_returns = []

    def update_alive(self, actor_id):
        self.actor_alive_time[actor_id] = time.time()

    def get_actor_alive_time_list(self):
        return self.actor_alive_time

    def add_return(self, g):
        self.actor_returns.append(g)

    def get_avg_return(self, n=1000):
        return np.mean(self.actor_returns[-n:])
