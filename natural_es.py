import torch
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
import numpy as np
from utils import *
import pickle
from config import *
import logging
import time

class Worker(mp.Process):
    def __init__(self, id, param, state_normalizer, task_q, result_q, stop, config):
        mp.Process.__init__(self)
        self.id = id
        self.task_q = task_q
        self.param = param
        self.result_q = result_q
        self.stop = stop
        self.config = config
        self.evaluator = Evaluator(config, state_normalizer)

    def run(self):
        config = self.config
        np.random.seed()
        while not self.stop.value:
            if self.task_q.empty():
                continue
            self.task_q.get()
            disturbed_param = np.copy(self.param.numpy().flatten())
            epsilon = np.random.randn(len(disturbed_param))
            disturbed_param += config.sigma * epsilon
            fitness, steps = self.evaluator.eval(disturbed_param)
            self.result_q.put([epsilon, -fitness, steps])

def train(config):
    task_queue = SimpleQueue()
    result_queue = SimpleQueue()
    stop = mp.Value('i', False)
    stats = SharedStats(config.state_dim)
    # if config.resume:
    #     with open('data/resume/%s-%s-best_solution.bin' % (config.tag, config.task), 'rb') as f:
    #         param = torch.FloatTensor(pickle.load(f))
    #     with open('data/resume/%s-%s-saved_shifter.bin' % (config.tag, config.task), 'rb') as f:
    #         stats.load_state_dict(pickle.load(f))
    # else:
    param = torch.FloatTensor(torch.from_numpy(config.initial_weight))
    param.share_memory_()
    normalizers = [StaticNormalizer(config.state_dim) for _ in range(config.num_workers)]
    for normalizer in normalizers:
        normalizer.offline_stats.load(stats)
    workers = [Worker(id, param, normalizers[id], task_queue, result_queue, stop, config) for id in range(config.num_workers)]
    for w in workers: w.start()

    training_rewards = []
    training_steps = []
    training_timestamps = []
    initial_time = time.time()
    total_steps = 0
    iteration = 0
    while not stop.value:
        if not total_steps:
            test_mean, test_ste = test(config, param.numpy(), stats)
            training_rewards.append(test_mean)
            training_steps.append(total_steps)
            training_timestamps.append(0)
            gym.logger.info('total steps %d, %f(%f)' % (total_steps, test_mean, test_ste))
        for i in range(config.popsize):
            task_queue.put(i)
        rewards = []
        epsilons = []
        steps = []
        while len(rewards) < config.popsize:
            if result_queue.empty():
                continue
            epsilon, fitness, step = result_queue.get()
            epsilons.append(epsilon)
            rewards.append(fitness)
            steps.append(step)

        total_steps += np.sum(steps)
        elapsed_time = time.time() - initial_time
        training_rewards.append(np.max(rewards))
        training_steps.append(total_steps)
        training_timestamps.append(elapsed_time)
        gym.logger.info('max rewards %f, total steps %d, elapsed time %f' %
                        (np.max(rewards), total_steps, elapsed_time))

        r_mean = np.mean(rewards)
        r_std = np.std(rewards)
        rewards = (rewards - r_mean) / r_std
        gym.logger.info('iteration %d, %f(%f)' % (iteration, r_mean, r_std / np.sqrt(config.popsize)))
        iteration += 1
        # if r_mean > config.target:
        if total_steps > config.max_steps:
            stop.value = True
            break
        for normalizer in normalizers:
            stats.merge(normalizer.online_stats)
            normalizer.online_stats.zero()
        for normalizer in normalizers:
            normalizer.offline_stats.load(stats)
        # with open('data/%s-%s-best_solution.bin' % (config.tag, config.task), 'wb') as f:
        #     pickle.dump(param.numpy(), f)
        # with open('data/%s-%s-saved_shifter.bin' % (config.tag, config.task), 'wb') as f:
        #     pickle.dump(stats.state_dict(), f)
        gradient = np.asarray(epsilons) * np.asarray(rewards).reshape((-1, 1))
        gradient = np.mean(gradient, 0) / config.sigma
        gradient = torch.FloatTensor(gradient)
        param.add_(config.learning_rate * gradient)

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
    return np.mean(rewards), np.std(rewards) / config.test_repetitions

def multi_runs(config):
    stats = []
    runs = 10
    for run in range(runs):
        gym.logger.info('Run %d' % (run))
        stats.append(train(config))
        with open('data/NES-stats-%s.bin' % (config.task), 'wb') as f:
            pickle.dump(stats, f)

if __name__ == '__main__':
    # config = PendulumConfig()
    # config.max_steps = int(1e8)

    # config = BipedalWalkerConfig()
    # config.max_steps = int(1e8)

    config = ContinuousLunarLanderConfig()
    config.max_steps = int(2e7)

    config.sigma = 0.1
    config.learning_rate = 1e-2
    config.resume = False
    config.test_repetitions = 5
    config.popsize = 64
    config.num_workers = 8

    fh = logging.FileHandler('log/NES-%s.txt' % config.task)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    gym.logger.addHandler(fh)

    # train(config)
    multi_runs(config)
