# Distributed Reinforcement Learning with Ape-X
## 1. Introduction
This repo contains the code of implementing the distributed reinforcement learning (RL) framework based on Ape-X.

This is the final project of the course (CS 7610): Foundation of Distributed System.

## 2. Prerequisites
In order to run the demo on your computer, we recommend you using the Anaconda virtual environment. Please make sure
you have already installed Anaconda.

### 2.1 Create the a virtual environment using Anaconda
Open a terminal, you can create an empty virtual environment using the following command:

`` conda create -n <your virtual env name> ``

Activate the virtual environment

`` conda activate <your virtual env name>``

### 2.2 Install the dependencies
You have to install the following dependencies to run the demo

```
1. python: conda install python
2. pytorch: please go to pytorch website and get the command
3. ray: conda install ray
4. gym: pip install gym
5. matplotlib: conda install matplotlib
6. numpy: conda install numpy
```

## 3. Run a distributed DQN demo on gym
Now, you are ready to run a demo of our implementation. In the activated conda environment, run the
following command: (Here we use CartPole-v0 as an example. --env=domain name, --worker_num=number of actors)

``python distributed_DQN_ray.py --env=CartPole-v0 --worker_num=2``

To plot the learning curve of the trained environment, please run:

``python plot_DQN_learning_curve.py --env=CartPole-v0``

To visualize the policy, please run:

``python visualize_DQN_policy.py --env=CartPole-v0``

Please note, you can try 3 different domains: CartPole-v0, CartPole-v1, and LundaLander-v2


## 4. Run a distributed DDPG demo on the gym Pendulum-v0 Env.
``python distributed_DDPG_ray.py``

Note, the default settings require at least 6 CPU cores.

## 5. Viewing Training Results
The saved performance plot from the most recent training run can be viewed using:

``python plot.py``

## Reference
1. [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)
2. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
3. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
4. [Ray](https://docs.ray.io/en/master/index.html)
5. [Pytorch](https://pytorch.org/)
6. [OpenAI Gym](https://gym.openai.com/)
