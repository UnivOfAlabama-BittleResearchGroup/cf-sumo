import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import itertools


def generate_initial_population(population_size, param_ranges):
    population = []
    for _ in range(population_size):
        individual = {
            key: np.random.uniform(low, high) for key, (low, high) in param_ranges.items()
        }
        population.append(individual)
    return population

def selection(population, fitnesses, num_parents):
    sorted_indices = np.argsort(fitnesses)
    parents = [population[i] for i in sorted_indices[:num_parents]]
    return parents

def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        offspring.append(child)
    return offspring

def mutation(offspring, param_ranges, mutation_rate):
    for child in offspring:
        for key, (low, high) in param_ranges.items():
            if random.random() < mutation_rate:
                child[key] = np.random.uniform(low, high)
    return offspring

def genetic_algorithm(population_size, num_generations, num_parents, param_ranges, simulation_func, mutation_rate=0.1):
    population = generate_initial_population(population_size, param_ranges)
    run_counter = {"count": 0}

    with ThreadPoolExecutor() as executor:
        for generation in range(num_generations):
            fitnesses = list(executor.map(simulation_func, itertools.repeat(run_counter, population_size), population))
            parents = selection(population, fitnesses, num_parents)
            offspring_size = population_size - len(parents)
            offspring = crossover(parents, offspring_size)
            offspring = mutation(offspring, param_ranges, mutation_rate)
            population = parents + offspring

    best_individual = min(population, key=lambda ind: simulation_func(run_counter, ind))
    return best_individual

