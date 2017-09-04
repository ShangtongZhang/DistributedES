import opensim as osim
from osim.env import RunEnv
import numpy as np
from model import *

class LTR:
    def __init__(self, difficulty=2, visualize=False):
        self.env = RunEnv(visualize=visualize)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.difficulty = difficulty

    def step(self, action):
        action = np.clip(action, 0, 1)
        next_state, reward, done, info = self.env.step(action)
        return np.asarray(next_state), reward, done, info

    def random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        state = self.env.reset(difficulty=self.difficulty, seed=np.random.randint(0, 10000000))
        return np.asarray(state)

class LearningToRunConfig:
    def __init__(self):
        self.task = 'LearningToRun'
        self.env_fn = lambda: LTR()
        self.repetitions = 3
        env = self.env_fn()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        model = SingleHiddenLayerNet(self.state_dim, self.action_dim)
        self.action_clip = lambda a: np.clip(a, 0, 1)
        self.initial_weight = model.get_weight()
        self.target = -50
        self.popsize = 100
        self.num_workers = 15