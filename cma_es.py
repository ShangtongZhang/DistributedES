import cma
import numpy as np
import gym
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
import pickle
import sys
from model import *
from utils import *
from config import *
import logging
import time

class Worker(mp.Process):
    def __init__(self, id, state_normalizer, task_q, result_q, stop, config):
        mp.Process.__init__(self)
        self.task_queue = task_q
        self.result_q = result_q
        self.evaluator = Evaluator(config, state_normalizer)
        self.id = id
        self.stop = stop

    def run(self):
        np.random.seed()
        while not self.stop.value:
            if self.task_queue.empty():
                continue
            id, solution = self.task_queue.get()
            fitness, steps = self.evaluator.eval(solution)
            # with open('data/%s-saved_shifter_%d.bin' % (TAG, self.id), 'wb') as f:
            #     pickle.dump(self.evaluator.shifter.state_dict(), f)
            self.result_q.put((id, fitness, steps))

def train(config):
    task_queue = SimpleQueue()
    result_queue = SimpleQueue()
    stop = mp.Value('i', False)
    stats = SharedStats(config.state_dim)
    normalizers = [StaticNormalizer(config.state_dim) for _ in range(config.num_workers)]
    for normalizer in normalizers:
        normalizer.offline_stats.load(stats)

    workers = [Worker(id, normalizers[id], task_queue, result_queue, stop, config) for id in range(config.num_workers)]
    for w in workers: w.start()

    opt = cma.CMAOptions()
    opt['tolfun'] = -config.target
    opt['popsize'] = config.pop_size
    opt['verb_disp'] = 1
    opt['verb_log'] = 0
    opt['maxiter'] = sys.maxsize
    es = cma.CMAEvolutionStrategy(config.initial_weight, 0.5, opt)

    total_steps = 0
    initial_time = time.time()
    training_rewards = []
    training_steps = []
    training_timestamps = []
    test_mean, test_ste = test(config, config.initial_weight, stats)
    gym.logger.info('total steps %d, %f(%f)' % (total_steps, test_mean, test_ste))
    training_rewards.append(test_mean)
    training_steps.append(0)
    training_timestamps.append(0)
    while True:
        solutions = es.ask()
        for id, solution in enumerate(solutions):
            task_queue.put((id, solution))
        while not task_queue.empty():
            continue
        result = []
        while len(result) < len(solutions):
            if result_queue.empty():
                continue
            result.append(result_queue.get())
        result = sorted(result, key=lambda x: x[0])
        total_steps += np.sum([r[2] for r in result])
        cost = [r[1] for r in result]
        best_solution = solutions[np.argmin(cost)]
        elapsed_time = time.time() - initial_time
        test_mean, test_ste = test(config, best_solution, stats)
        gym.logger.info('total steps %d, test %f(%f), best %f, elapased time %f' %
                        (total_steps, test_mean, test_ste, np.argmin(cost), elapsed_time))
        training_rewards.append(test_mean)
        training_steps.append(total_steps)
        training_timestamps.append(elapsed_time)
        # with open('data/%s-best_solution_%s.bin' % (TAG, config.task), 'wb') as f:
        #     pickle.dump(solutions[np.argmin(result)], f)
        if config.max_steps and total_steps > config.max_steps:
            stop.value = True
            break
        es.tell(solutions, cost)
        # es.disp()
        for normalizer in normalizers:
            stats.merge(normalizer.online_stats)
            normalizer.online_stats.zero()
        for normalizer in normalizers:
            normalizer.offline_stats.load(stats)

    stop.value = True
    for w in workers: w.join()
    return [training_rewards, training_steps, training_timestamps]

def test(config, solution, stats):
    normalizer = StaticNormalizer(config.state_dim)
    normalizer.offline_stats.load_state_dict(stats.state_dict())
    evaluator = Evaluator(config, normalizer)
    evaluator.model.set_weight(solution)
    rewards = []
    for i in range(config.test_repetitions):
        reward, _ = evaluator.single_run()
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards) / config.repetitions

def multi_runs(config):
    fh = logging.FileHandler('log/CMA-%s.txt' % config.task)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    gym.logger.addHandler(fh)

    stats = []
    runs = 10
    for run in range(runs):
        gym.logger.info('Run %d' % (run))
        stats.append(train(config))
        with open('data/CMA-stats-%s.bin' % (config.task), 'wb') as f:
            pickle.dump(stats, f)

def all_tasks():
    configs = []

    config = PendulumConfig()
    config.max_steps = int(4e7)
    configs.append(config)

    config = BipedalWalkerConfig()
    config.max_steps = int(2e8)
    configs.append(config)

    config = ContinuousLunarLanderConfig()
    config.max_steps = int(4e7)
    configs.append(config)

    ps = []
    for cf in configs:
        # cf.max_steps = int(1e10)
        cf.num_workers = 8
        cf.pop_size = 64
        ps.append(mp.Process(target=multi_runs, args=(cf, )))

    for p in ps: p.start()
    for p in ps: p.join()

if __name__ == '__main__':
    # configs = []
    #
    # config = PendulumConfig()
    # config.max_steps = int(1e8)
    # configs.append(config)
    #
    # config = BipedalWalkerConfig()
    # config.max_steps = int(1e8)
    # configs.append(config)
    #
    # config = ContinuousLunarLanderConfig()
    # config.max_steps = int(2e7)
    # configs.append(config)
    #
    # for config in configs:
    #     config.max_steps = int(1e10)
    #     config.num_workers = 8
    #     config.pop_size = 64
    #
    # config = configs[0]
    # fh = logging.FileHandler('log/NES-%s.txt' % config.task)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # fh.setLevel(logging.DEBUG)
    # gym.logger.addHandler(fh)

    # train(config)
    # all_tasks()
    config = BipedalWalkerHardcore()
    config.max_steps = int(2e8)
    multi_runs(config)
