import ray

ray.init()  # init ray on the local machine
assert ray.is_initialized(), "Ray fails to initialize."

"""
    Simple classes for basic definition and remote calls
"""
@ray.remote
class Actor(object):
    def __init__(self, actor_id, init_val):
        self.id = actor_id
        self.value = init_val

    def increase(self, add_val):
        self.value += add_val

    def get_value(self):
        return self.value

    def get_id(self):
        return self.id

    @ray.method(num_returns=2)
    def get_all(self):
        return self.id, self.value


@ray.remote
class Sum(object):
    def __init__(self, actors_list):
        self.actors = actors_list
        self.sum = 0

    def add_actors(self):
        for actor_item in self.actors:
            self.sum += ray.get(actor_item.get_value.remote())

    def get_sum_val(self):
        return self.sum


if __name__ == '__main__':
    # create actors
    actor_num = 5
    actors = [Actor.remote(i, i) for i in range(actor_num)]
    sum_actor = Sum.remote(actors)
    sum_actor.add_actors.remote()

    print(ray.get(sum_actor.get_sum_val.remote()))

    # test single actor
    actor = Actor.remote(0, 0)
    obj1_id, obj2_val = actor.get_all.remote()

    print(ray.get(obj1_id), ' - ', ray.get(obj2_val))

    ray.shutdown()