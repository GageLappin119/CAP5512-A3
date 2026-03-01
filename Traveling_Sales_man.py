import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import math

import itertools

POPULATION_SIZE = 300
MUTATION_RATE = 0.15
MAX_GEN = 500
CROSSOVER_RATE = 0.85
NUM_RUNS = 100
TOURNAMENT_SIZE = 3
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

DIST_MATRIX = np.zeros((49, 49))
for i in range(49):
    for j in range(49):
        DIST_MATRIX[i][j] = haversine_distance(i, j)

def create_individual():
    cities = list(range(49))

    random.shuffle(cities)

    return cities

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

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
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

def run_one_experiment(run_id, pop_size, max_gen, cx_rate, mut_rate):
    random.seed(run_id)
    pop = toolbox.population(n=pop_size)

    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    best_fitness_history = []
    avg_fitness_history = []
    optimum_found_gen = None

    for g in range(max_gen):
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
            if random.random() < cx_rate: 
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < mut_rate:
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
def plot_route(best_ind, data_list):
    # Generates and saves a map of the route based on Lats and Lons
    lons = []
    lats = []
    
    for idx in best_ind:
        city, lat, lon = data_list[idx]
        lons.append(lon)
        lats.append(lat)
        
    lons.append(lons[0])
    lats.append(lats[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(lons, lats, marker='o', linestyle='-', color='b', markersize=5)
    
    plt.plot(lons[0], lats[0], marker='s', color='red', markersize=8, label='Start/End')
    
    plt.title("Best 48-City TSP Route Found")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsp_route_map.png")
    plt.show()

def plot_convergence(b_hist, a_hist):
    # Generates and saves the fitness convergence graph
    generations = range(len(b_hist))
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, b_hist, label="Best Distance", color='red', linewidth=2)
    plt.plot(generations, a_hist, label="Average Distance", color='blue', alpha=0.5)
    
    plt.title("GA Convergence over 500 Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance (Miles)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_plot.png") 
    plt.show()

def main():
    global DIST_MATRIX

    print(f"Starting {NUM_RUNS} runs...")
    
    pop_sizes = [500]
    mut_rates = [0.15]
    cx_rates = [0.7]
    
    # Left over from previous grid search

    test_configs = list(itertools.product(pop_sizes, mut_rates, cx_rates))

    for p, m, c in test_configs:
        print(f"\nTesting Config: Pop={p}, Mut={m}, CX={c}")
        results = []
        
        overall_best_dist = float('inf')
        overall_best_ind = None
        overall_best_b_hist = None
        overall_best_a_hist = None

        for i in range(NUM_RUNS):
            b_hist, a_hist, opt_gen, best_ind = run_one_experiment(i, p, 500, c, m)
            
            current_dist = best_ind.fitness.values[0]
            results.append(current_dist)
            
            if current_dist < overall_best_dist:
                overall_best_dist = current_dist
                overall_best_ind = best_ind
                overall_best_b_hist = b_hist
                overall_best_a_hist = a_hist
                
            if (i+1) % 10 == 0:
                print(f"Run {i+1}/{NUM_RUNS} completed.")
                
        avg_of_runs = sum(results) / len(results)
        print(f"Average Best Fitness for this config: {avg_of_runs:.2f}")
        print(f"Absolute Best Fitness Found: {overall_best_dist:.2f}")

    print("\nBest Route Discovered:")
    best_route_cities = [data_list[idx][0] for idx in overall_best_ind]
    print(" -> ".join(best_route_cities) + " -> " + best_route_cities[0])
    
    print("\nGenerating charts...")
    plot_route(overall_best_ind, data_list)
    plot_convergence(overall_best_b_hist, overall_best_a_hist)

if __name__ == "__main__":
    main()

