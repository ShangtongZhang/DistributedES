import numpy as np
import torch

class Shifter:
    def __init__(self, filter_mean=True):
        self.m = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def state_dict(self):
        return {'m': self.m,
                'v': self.v,
                'n': self.n}

    def load_state_dict(self, saved):
        self.m = saved['m']
        self.v = saved['v']
        self.n = saved['n']

    def __call__(self, o):
        self.m = self.m * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6) ** .5  # std
        self.n += 1
        if self.filter_mean:
            o_ = (o - self.m) / self.std
        else:
            o_ = o / self.std
        return o_

class StaticShifter:
    def __init__(self):
        self.m = 0
        self.v = 0

        self.n = 0



    def __call__(self, o):
        self.std = (self.v + 1e-6) ** .5
        return (o - self.m) / self.std

class SharedShifter:
    def __init__(self, o_size):
        self.m = torch.zeros(o_size)
        self.v = torch.zeros(o_size)
        self.n = torch.zeros(1)





class Evaluator:
    def __init__(self, config, shifter=None):
        self.model = config.model_fn()
        self.repetitions = config.repetitions
        self.env = config.env_fn()
        if shifter is None:
            self.shifter = Shifter()
        else:
            self.shifter = shifter
        self.config = config

    def eval(self, solution):
        self.model.set_weight(solution)
        rewards = []
        for i in range(self.repetitions):
            rewards.append(self.single_run())
        return -np.mean(rewards)

    def single_run(self):
        state = self.env.reset()
        total_reward = 0
        while True:
            state = self.shifter(state)
            action = self.model(np.stack([state])).data.numpy().flatten()
            action = self.config.action_clip(action)
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                return total_reward