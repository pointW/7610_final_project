class ParamServer(object):
    def __init__(self, model_state_dict):
        self.model_state_dict = model_state_dict  # parameters of the model
        self.eps = 1  # epsilon for exploration

    def sync_learner_model_params(self, model_stat_dict, eps):  # synchronize the parameters with the learner
        self.model_state_dict = model_stat_dict
        self.eps = eps

    def get_latest_model_params(self):  # return the latest model parameters
        return self.model_state_dict

    def get_scheduled_eps(self):  # return the latest epsilon
        return self.eps