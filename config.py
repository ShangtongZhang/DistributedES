import gym
from model import *

class PendulumConfig:
    def __init__(self):
        self.task = 'Pendulum-v0'
        self.env_fn = lambda: gym.make(self.task)
        self.repetitions = 5
        env = self.env_fn()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.model_fn = lambda: StandardFCNet(self.state_dim, self.action_dim)
        model = self.model_fn()
        self.action_clip = lambda a: np.clip(a, -2, 2)
        self.initial_weight = model.get_weight()
        self.target = 0
        self.popsize = 30
        self.num_workers = 6

class BipedalWalkerConfig:
    def __init__(self):
        self.task = 'BipedalWalker-v2'
        self.env_fn = lambda: gym.make(self.task)
        self.repetitions = 5
        env = self.env_fn()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.model_fn = lambda: StandardFCNet(self.state_dim, self.action_dim)
        model = self.model_fn()
        self.action_clip = lambda a: np.clip(a, -1, 1)
        self.initial_weight = model.get_weight()
        self.target = 1000
        self.popsize = 30
        self.num_workers = 6

class ContinuousLunarLanderConfig:
    def __init__(self):
        self.task = 'LunarLanderContinuous-v2'
        self.env_fn = lambda: gym.make(self.task)
        self.repetitions = 5
        env = self.env_fn()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.model_fn = lambda: StandardFCNet(self.state_dim, self.action_dim)
        model = self.model_fn()
        self.action_clip = lambda a: np.clip(a, -1, 1)
        self.initial_weight = model.get_weight()
        self.target = 300
        self.popsize = 30
        self.num_workers = 6