import torch
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
import numpy as np
from utils import *
import pickle
from config import *
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
        test_mean, test_ste = test(config, param.numpy(), stats)
        elapsed_time = time.time() - initial_time
        training_rewards.append(test_mean)
        training_steps.append(total_steps)
        training_timestamps.append(elapsed_time)
        logger.info('Test: total steps %d, %f(%f), elapsed time %d' %
            (total_steps, test_mean, test_ste, elapsed_time))
        
        for i in range(config.pop_size):
            task_queue.put(i)
        rewards = []
        epsilons = []
        steps = []
        while len(rewards) < config.pop_size:
            if result_queue.empty():
                continue
            epsilon, fitness, step = result_queue.get()
            epsilons.append(epsilon)
            rewards.append(fitness)
            steps.append(step)

        total_steps += np.sum(steps)
        r_mean = np.mean(rewards)
        r_std = np.std(rewards)
        # rewards = (rewards - r_mean) / r_std
        logger.info('Train: iteration %d, %f(%f)' % (iteration, r_mean, r_std / np.sqrt(config.pop_size)))
        iteration += 1
        # if r_mean > config.target:
        if config.max_steps and total_steps > config.max_steps:
            stop.value = True
            break
        for normalizer in normalizers:
            stats.merge(normalizer.online_stats)
            normalizer.online_stats.zero()
        for normalizer in normalizers:
            normalizer.offline_stats.load(stats)
        rewards = fitness_shift(rewards)
        gradient = np.asarray(epsilons) * np.asarray(rewards).reshape((-1, 1))
        gradient = np.mean(gradient, 0) / config.sigma
        gradient = config.opt.update(gradient)
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
    fh = logging.FileHandler('log/NES-%s.txt' % config.task)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    stats = []
    runs = 10
    for run in range(runs):
        logger.info('Run %d' % (run))
        stats.append(train(config))
        with open('data/NES-stats-%s.bin' % (config.task), 'wb') as f:
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
        cf.sigma = 0.1
        cf.learning_rate = 1e-2
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
    #     config.sigma = 0.1
    #     config.learning_rate = 1e-2
    #
    # config = configs[0]
    # fh = logging.FileHandler('log/NES-%s.txt' % config.task)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    # all_tasks()
    # train(config)
    # multi_runs(config)
    # config = BipedalWalkerHardcore()

    config = PendulumConfig()
    config.max_steps = int(2e8)
    config.sigma = 0.1
    config.learning_rate = 1e-2
    multi_runs(config)
