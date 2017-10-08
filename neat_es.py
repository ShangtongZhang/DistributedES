import numpy as np
import gym
import math
import os
import pickle
import neat
from itertools import product
from neat.reporting import *
import logging
import time
import torch.multiprocessing as mp
from utils import *
from config import *
from neat.six_util import iteritems

class GenomeEvaluator:
    def __init__(self, config, neat_config, state_normalizer):
        self.config = config
        self.neat_config = neat_config
        self.state_normalizer = state_normalizer
        self.env = config.env_fn()
        self.repetitions = config.repetitions

    def eval_genome(self, genome):
        net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
        fitness = np.zeros(self.repetitions)
        steps = np.zeros(fitness.shape)
        for run in range(self.repetitions):
            fitness[run], steps[run] = self.evaluate(net)
        return np.mean(fitness), np.sum(steps)

    def evaluate(self, net):
        state = self.state_normalizer(self.env.reset())
        steps = 0
        rewards = 0
        while True:
            action = self.config.action_clip(net.activate(state))
            state, reward, done, _ = self.env.step(action)
            state = self.state_normalizer(state)
            rewards += reward
            steps += 1
            if done:
                break
        return self.config.reward_to_fitness(rewards), steps

class Worker(mp.Process):
    def __init__(self, id, state_normalizer, task_q, result_q, stop, config, neat_config):
        mp.Process.__init__(self)
        self.id = id
        self.task_q = task_q
        self.result_q = result_q
        self.state_normalizer = state_normalizer
        self.stop = stop
        self.config = config
        self.env = config.env_fn()
        self.evaluator = GenomeEvaluator(config, neat_config, state_normalizer)

    def run(self):
        np.random.seed()
        while not self.stop.value:
            if self.task_q.empty():
                continue
            id, genome = self.task_q.get()
            fitness, steps = self.evaluator.eval_genome(genome)
            self.result_q.put([id, fitness, steps])

class NEATAgent:
    def __init__(self, config):
        self.config = config
        self.neat_config = self.load_neat_config()
        self.neat_config.pop_size = config.popsize
        self.task_q = mp.SimpleQueue()
        self.result_q = mp.SimpleQueue()
        self.total_steps = 0
        stop = mp.Value('i', False)
        stats = SharedStats(config.state_dim)
        normalizers = [StaticNormalizer(config.state_dim) for _ in range(config.num_workers)]
        for normalizer in normalizers:
            normalizer.offline_stats.load(stats)
        workers = [Worker(id, normalizers[id], self.task_q, self.result_q, stop,
                          config, self.neat_config) for id in range(config.num_workers)]
        for w in workers: w.start()
        self.normalizers = normalizers
        self.stats = stats
        self.stop = stop

    def load_neat_config(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'neat-config/%s.txt' % self.config.task)
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        neat_config.fitness_threshold = self.config.target
        return neat_config

    def evaluate(self, genomes, _):
        tasks = [genome for _, genome in genomes]
        for id, task in enumerate(tasks):
            self.task_q.put([id, task])
        steps = 0
        results = []
        while len(results) < len(tasks):
            if self.result_q.empty():
                continue
            id, fitness, step = self.result_q.get()
            steps += step
            tasks[id].fitness = fitness
            results.append([id, fitness])
        for normalizer in self.normalizers:
            self.stats.merge(normalizer.online_stats)
            normalizer.online_stats.zero()
        for normalizer in self.normalizers:
            normalizer.offline_stats.load(self.stats)
        # best_fitness_ind = np.argmax([f for _, f in results])
        # best_genome_ind = results[best_fitness_ind][0]
        # best_genome = tasks[best_genome_ind]
        # fitness, _ = self.test(best_genome)
        self.total_steps += steps

    def test(self, genome):
        normalizer = StaticNormalizer(self.config.state_dim)
        normalizer.offline_stats.load(self.stats)
        evaluator = GenomeEvaluator(self.config, self.neat_config, normalizer)
        evaluator.repetitions = self.config.test_repetitions
        return evaluator.eval_genome(genome)

    def evolve(self):
        class CustomReporter(BaseReporter):
            def __init__(self, agent):
                self.fitness = []
                self.agent = agent
            def post_evaluate(self, config, population, species, best_genome):
                self.fitness.append(best_genome.fitness)
                print 'total steps %d, best fitness %f' % (self.agent.total_steps, best_genome.fitness)
                if best_genome.fitness > self.agent.config.target:
                    self.agent.stop.value = True

        pop = neat.Population(self.neat_config)
        # stats = neat.StatisticsReporter()
        # pop.add_reporter(stats)
        # pop.add_reporter(neat.StdOutReporter(True))
        reporter = CustomReporter(self)
        pop.add_reporter(reporter)
        pop.run(self.evaluate)

    def run(self):
        self.evolve()

if __name__ == '__main__':
    # config = PendulumConfig()
    # config.action_clip = lambda a: 2 * a

    # config = ContinuousLunarLanderConfig()

    config = BipedalWalkerConfig()

    config.popsize = 100
    config.test_repetitions = 5
    config.reward_to_fitness = lambda r: r
    NEATAgent(config).run()
