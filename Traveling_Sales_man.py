import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import math
import csv
import multiprocessing

POPULATION_SIZE = 750
MUTATION_RATE = 0.2
MAX_GEN = 1000
CROSSOVER_RATE = 0.7
NUM_RUNS = 50
DIST_MATRIX = []

def haversine_distance(index_1, index_2):
    city_1 = data_list[index_1]
    city_2 = data_list[index_2]

    lat1, lon1 = city_1[1], city_1[2]
    lat2, lon2 = city_2[1], city_2[2]

    R = 3958.8

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

data_list = []

file_path = 'tsp.dat'

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
            
        parts = line.rsplit(None, 2)
        
        if len(parts) == 3:
            city = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            data_list.append((city, lat, lon))

DIST_MATRIX = np.zeros((48, 48))
for i in range(48):
    for j in range(48):
        DIST_MATRIX[i][j] = haversine_distance(i, j)

def create_individual():
    cities = list(range(48))

    random.shuffle(cities)

    return cities

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate_distances(individual):
    running_distance = 0
    start_capital = individual[0]

    for i in range(len(individual) - 1):
        running_distance += haversine_distance(individual[i], individual[i + 1])
    
    running_distance += haversine_distance(start_capital, individual[len(individual) - 1])
    return running_distance,

def evaluate_distances(individual):
    dist = 0
    for i in range(len(individual) - 1):
        dist += DIST_MATRIX[individual[i]][individual[i+1]]
    dist += DIST_MATRIX[individual[0]][individual[len(individual) - 1]]
    return dist,

def reverse_list(individual, indpb):
    if random.random() < indpb:
        individual[0].reverse()

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_distances)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutInversion)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_one_experiment(run_id):
    random.seed(run_id)
    pop = toolbox.population(n=POPULATION_SIZE)

    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    best_fitness_history = []
    avg_fitness_history = []
    optimum_found_gen = None

    for g in range(MAX_GEN):
        fits = [ind.fitness.values[0] for ind in pop]
        current_best = min(fits)
        current_avg = sum(fits) / len(pop)

        best_fitness_history.append(current_best)
        avg_fitness_history.append(current_avg)

        elite = tools.selBest(pop, 1)[0]
        elite_clone = toolbox.clone(elite)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE: 
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        pop[0] = elite_clone
    best_ind = tools.selBest(pop, 1)[0]
    return best_fitness_history, avg_fitness_history, optimum_found_gen, best_ind

def main():
    global DIST_MATRIX

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print(f"Starting {NUM_RUNS} runs...")
    
    all_best_fitness = [] 
    all_avg_fitness = []
    all_optimum_gens = []
    global_bests = []
    best_ind = 0

    for i in range(NUM_RUNS):
        b_hist, a_hist, opt_gen, best_ind = run_one_experiment(i)
        
        all_best_fitness.append(b_hist)
        all_avg_fitness.append(a_hist)
        
        global_bests.append(min(b_hist))
        
        if opt_gen is not None:
            all_optimum_gens.append(opt_gen)
            
        if (i+1) % 10 == 0:
            print(f"Run {i+1}/{NUM_RUNS} completed.")
    
    print(evaluate_distances(best_ind))

if __name__ == "__main__":
    main()