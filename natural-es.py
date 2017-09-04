import torch
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
import numpy as np
from utils import *
import pickle
from config import *
import sys

class Worker(mp.Process):
    def __init__(self, id, param, task_q , result_q, stop, config):
        mp.Process.__init__(self)
        self.id = id
        self.task_q = task_q
        self.param = param
        self.result_q = result_q
        self.stop = stop
        self.config = config
        self.evaluator = Evaluator(config)
        if config.resume:
            with open('data/resume/%s-%s-saved_shifter_%d.bin' % (config.tag, config.task, self.id), 'rb') as f:
                self.evaluator.shifter.load_state_dict(pickle.load(f))

    def run(self):
        np.random.seed()
        while not self.stop.value:
            if self.task_q.empty():
                continue
            self.task_q.get()
            disturbed_param = np.copy(self.param.numpy().flatten())
            epsilon = np.random.randn(len(disturbed_param))
            disturbed_param += self.config.sigma * epsilon
            fitness = self.evaluator.eval(disturbed_param)
            with open('data/%s-%s-saved_shifter_%d.bin' % (config.tag, config.task, self.id), 'wb') as f:
                pickle.dump(self.evaluator.shifter.state_dict(), f)
            self.result_q.put([epsilon, -fitness])

def train(config):
    task_queue = SimpleQueue()
    result_queue = SimpleQueue()
    stop = mp.Value('i', False)
    if config.resume:
        with open('data/resume/%s-%s-best_solution.bin' % (config.tag, config.task), 'rb') as f:
            param = torch.FloatTensor(pickle.load(f))
    else:
        param = torch.FloatTensor(torch.from_numpy(config.initial_weight))
    param.share_memory_()
    workers = [Worker(id, param, task_queue, result_queue, stop, config) for id in range(config.num_workers)]
    for w in workers: w.start()

    iteration = 0
    while not stop.value:
        with open('data/%s-%s-best_solution.bin' % (config.tag, config.task), 'wb') as f:
            pickle.dump(param.numpy(), f)
        for i in range(config.popsize):
            task_queue.put(i)
        rewards = []
        epsilons = []
        while len(rewards) < config.popsize:
            if result_queue.empty():
                continue
            epsilon, fitness = result_queue.get()
            epsilons.append(epsilon)
            rewards.append(fitness)
        r_mean = np.mean(rewards)
        r_std = np.std(rewards)
        rewards = (rewards - r_mean) / r_std
        print iteration, r_mean, r_std / np.sqrt(config.popsize)
        iteration += 1
        if r_mean > config.target:
            stop.value = True
            break
        gradient = np.asarray(epsilons) * np.asarray(rewards).reshape((-1, 1))
        gradient = np.mean(gradient, 0) / config.sigma
        gradient = torch.FloatTensor(gradient)
        param.add_(config.learning_rate * gradient)

    for w in workers: w.join()

def test(id, config):
    with open('data/%s-%s-best_solution.bin' % (config.tag, config.task), 'rb') as f:
        solution = pickle.load(f)
    with open('data/%s-%s-saved_shifter_%d.bin' % (config.tag, config.task, id), 'rb') as f:
        saved_shifter = pickle.load(f)

    evaluator = Evaluator(config)
    evaluator.shifter.load_state_dict(saved_shifter)
    evaluator.model.set_weight(solution)
    rewards = []
    repetitions = 10
    for i in range(repetitions):
        rewards.append(evaluator.single_run())
    print rewards
    print np.mean(rewards), np.std(rewards) / repetitions

if __name__ == '__main__':
    config = PendulumConfig()
    # config = BipedalWalkerConfig()
    config.sigma = 0.1
    config.learning_rate = 1e-2
    config.tag = 'NES'
    config.resume = False

    train(config)
    # test(0, config)
