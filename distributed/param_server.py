class ParamServer(object):
    def __init__(self, model_state_dict, total_train_steps):
        self.model_state_dict = model_state_dict  # parameters of the model
        self.eps = 1  # epsilon for exploration
        self.total_train_steps = total_train_steps
        self.current_train_step = 0

    def sync_learner_model_params(self, model_stat_dict, eps, step):  # synchronize the parameters with the learner
        self.model_state_dict = model_stat_dict
        self.eps = eps
        self.current_train_step = step

    def get_latest_model_params(self):  # return the latest model parameters
        return self.model_state_dict

    def get_scheduled_eps(self):  # return the latest epsilon
        return self.eps

    def get_current_train_step(self):  # return the current train step
        return self.current_train_step

    def get_total_train_steps(self):
        return self.total_train_steps  # return the total train steps

