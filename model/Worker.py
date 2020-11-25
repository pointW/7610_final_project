import ray
import gym
import os
ray.init()


@ray.remote  # specify the class will act as an actor in a different process
class RolloutWorker(object):
    """
        Ideally the work should send back the transition when it is ready
    """
    def __init__(self, env_params):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = gym.make(env_params['env_name'])
        self.episode_time_steps = env_params['max_episode_time_steps']

    def rollout(self, agent_model):
        # reset the environment
        obs, rewards = self.env.reset(), []

        # transition
        transitions = []

        # perform the rollout
        for i in range(self.episode_time_steps):
            # get one action
            action = agent_model.get_action(obs)
            # step one action
            next_obs, reward, done, _ = self.env.step(action)

            # store the transitions
            transitions.append((obs, action, reward, next_obs, done))

            # check termination
            if done:
                break
            else:
                obs = next_obs

        return transitions



