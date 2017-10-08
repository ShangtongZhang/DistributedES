import cma
import numpy as np
import gym
import multiprocessing
from multiprocessing.queues import SimpleQueue
import pickle
import sys
from model import *
from utils import *
from config import *

# TAG = 'cma-es'

class Worker(multiprocessing.Process):
    def __init__(self, id, state_normalizer, task_q, result_q, stop, config):
        multiprocessing.Process.__init__(self)
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
    stop = multiprocessing.Value('i', False)
    stats = SharedStats(config.state_dim)
    normalizers = [StaticNormalizer(config.state_dim) for _ in range(config.num_workers)]
    for normalizer in normalizers:
        normalizer.offline_stats.load(stats)

    workers = [Worker(id, normalizers[id], task_queue, result_queue, stop, config) for id in range(config.num_workers)]
    for w in workers: w.start()

    opt = cma.CMAOptions()
    opt['tolfun'] = -config.target
    opt['popsize'] = config.popsize
    opt['verb_disp'] = 1
    opt['verb_log'] = 0
    opt['maxiter'] = sys.maxsize
    es = cma.CMAEvolutionStrategy(config.initial_weight, 0.5, opt)

    total_steps = 0
    test_mean, test_ste = test(config, config.initial_weight, stats)
    print 'total steps %d, %f(%f)' % (total_steps, test_mean, test_ste)
    while not es.stop():
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
        test_mean, test_ste = test(config, best_solution, stats)
        print 'total steps %d, %f(%f)' % (total_steps, test_mean, test_ste)
        # with open('data/%s-best_solution_%s.bin' % (TAG, config.task), 'wb') as f:
        #     pickle.dump(solutions[np.argmin(result)], f)
        es.tell(solutions, cost)
        es.disp()
        for normalizer in normalizers:
            stats.merge(normalizer.online_stats)
            normalizer.online_stats.zero()
        for normalizer in normalizers:
            normalizer.offline_stats.load(stats)

    stop.value = True
    for w in workers: w.join()

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

if __name__ == '__main__':
    config = PendulumConfig()
    # config = BipedalWalkerConfig()
    # config = ContinuousLunarLanderConfig()
    config.test_repetitions = 5
    train(config)
