import time


class ActorStateServer:
    def __init__(self, actor_num):
        self.actor_alive_time = [time.time() for _ in range(actor_num)]

    def update_alive(self, actor_id):
        self.actor_alive_time[actor_id] = time.time()

    def get_actor_alive_time_list(self):
        return self.actor_alive_time
