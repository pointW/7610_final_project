from torch import nn



# class of deep neural network model
class DeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim):
        # inherit from nn module
        super(DeepQNet, self).__init__()
        # feed forward network
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Identity()
        )

    # forward function
    def forward(self, state):
        x = self.fc_layer(state)
        return x

#DDPG Pendulum settings
class Actor(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim, output_activation=None):
        # inherit from nn module
        super(Actor, self).__init__()
        # feed forward network
        if output_activation is None:
            output_activation = nn.Identity()

        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            output_activation
        )

    # forward function
    def forward(self, state):
        x = self.fc_layer(state)
        return x

class Critic(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim):
        # inherit from nn module
        super(Critic, self).__init__()
        # feed forward network

        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Identity()
        )

    # forward function
    def forward(self, state):
        x = self.fc_layer(state)
        return x
