import gym
from model import *
from utils import *

class BasicConfig:
    def __init__(self, hidden_size):
        self.env_fn = lambda: gym.make(self.task)
        self.repetitions = 10
        self.test_repetitions = 10
        env = self.env_fn()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.hidden_size = hidden_size
        self.model_fn = lambda: StandardFCNet(self.state_dim, self.action_dim, self.hidden_size)
        model = self.model_fn()
        self.initial_weight = model.get_weight()
        self.reward_to_fitness = lambda r: r
        self.pop_size = 30
        self.num_workers = 6
        self.max_steps = 0
        self.opt = Adam()
        self.weight_decay = 0.005
        self.action_noise_std = 0
        self.tag = ''

class PendulumConfig(BasicConfig):
    def __init__(self, hidden_size):
        self.task = 'Pendulum-v0'
        self.action_clip = lambda a: np.clip(a, -2, 2)
        self.target = 10000
        BasicConfig.__init__(self, hidden_size)


class BipedalWalkerConfig(BasicConfig):
    def __init__(self, hidden_size):
        self.task = 'BipedalWalker-v2'
        self.action_clip = lambda a: np.clip(a, -1, 1)
        self.target = 10000
        BasicConfig.__init__(self, hidden_size)

class ContinuousLunarLanderConfig(BasicConfig):
    def __init__(self, hidden_size):
        self.task = 'LunarLanderContinuous-v2'
        self.action_clip = lambda a: np.clip(a, -1, 1)
        self.target = 10000
        BasicConfig.__init__(self, hidden_size)

class BipedalWalkerHardcore(BasicConfig):
    def __init__(self, hidden_size):
        self.task = 'BipedalWalkerHardcore-v2'
        self.action_clip = lambda a: np.clip(a, -1, 1)
        self.target = 10000
        BasicConfig.__init__(self, hidden_size)