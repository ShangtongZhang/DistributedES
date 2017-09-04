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

TAG = 'cma-es'

class Worker(multiprocessing.Process):
    def __init__(self, id, task_q, result_q, stop, config):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_q
        self.result_q = result_q
        self.evaluator = Evaluator(config)
        self.id = id
        self.stop = stop

    def run(self):
        while not self.stop.value:
            if self.task_queue.empty():
                continue
            id, solution = self.task_queue.get()
            fitness = self.evaluator.eval(solution)
            with open('data/%s-saved_shifter_%d.bin' % (TAG, self.id), 'wb') as f:
                pickle.dump(self.evaluator.shifter.state_dict(), f)
            self.result_q.put((id, fitness))

def train(config):
    task_queue = SimpleQueue()
    result_queue = SimpleQueue()
    stop = multiprocessing.Value('i', False)
    workers = [Worker(id, task_queue, result_queue, stop, config) for id in range(config.num_workers)]
    for w in workers: w.start()

    opt = cma.CMAOptions()
    opt['tolfun'] = -config.target
    opt['popsize'] = config.popsize
    opt['verb_disp'] = 1
    opt['verb_log'] = 0
    opt['maxiter'] = sys.maxsize
    es = cma.CMAEvolutionStrategy(config.initial_weight, 0.5, opt)

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
        result = [r[1] for r in result]
        with open('data/%s-best_solution_%s.bin' % (TAG, config.task), 'wb') as f:
            pickle.dump(solutions[np.argmin(result)], f)
        es.tell(solutions, result)
        es.disp()

    stop.value = True
    for w in workers: w.join()

def test(id, config):
    with open('data/%s-best_solution_%s.bin' % (TAG, config.task), 'rb') as f:
        solution = pickle.load(f)
    with open('data/%s-saved_shifter_%d.bin' % (TAG, id), 'rb') as f:
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
    # config = LearningToRunConfig()
    train(config)
    # test(0)
