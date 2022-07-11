import os, sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import random
from TSPData import TSPData

# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size
        self.pc = 0.7
        self.pm = 0.01

     # Knuth-Yates shuffle, reordering a array randomly
     # @param chromosome array to shuffle.
    def shuffle(self, chromosome):
        n = len(chromosome)
        for i in range(n):
            r = i + int(random.uniform(0, 1) * (n - i))
            swap = chromosome[r]
            chromosome[r] = chromosome[i]
            chromosome[i] = swap
        return chromosome

    # Fitness function for the GA, calculating shortest distance from start to end through the specified route
    # @param chromosome array indicating the order of visiting product locations
    # @param product_to_product list containing distances between product locations
    # @param start_to_product list containing distances from start to product locations
    # @param product_to_end list containing distances from product locations to end
    def fitness(self, chromosome, product_to_product, start_to_product, product_to_end):
        result = start_to_product[chromosome[0]]
        for i in range(len(chromosome) - 1):
            result = result + product_to_product[chromosome[i]][chromosome[i + 1]]

        result = result + product_to_end[chromosome[len(chromosome) - 1]] + len(chromosome)

        return result

    # Creates the initial population by creating all candidate solutions and then selecting randomly K solutions out of them
    # @param no_products - total number of products
    def create_initial_population(self, no_products):
        list = []
        for i in range(no_products):
            list.append(i)
        candidate_solutions = []
        solution = list
        candidate_solutions.append(solution)
        no_solutions = np.power(no_products, 5)   # total number of possible solutions

        # create all candidate solutions
        while len(candidate_solutions) < no_solutions:
            solution = self.shuffle(solution)
            temp = solution.copy()
            candidate_solutions.append(temp)

        # select K solutions for the initial population
        population = random.sample(candidate_solutions, self.pop_size)
        return population

    # Mutates chromosome by swapping 2 genes, if the probability is not met then the original chromosome is returned
    def mutate(self, chromosome):
        chance = random.random()
        if chance <= self.pm:
            i = random.randint(0, len(chromosome) - 1)
            pos2 = random.randint(0, len(chromosome) - 1)
            while pos2 == i:
                pos2 = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[pos2] = chromosome[pos2], chromosome[i]

        return chromosome

    # Performs the cross-over between the parents and returns the new children
    # @param parent1 - First parent chromosome
    # @param parent2 - Second parent chromosome
    def cross_over(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Choose at which gene (index) to start the cross-over
        start = 0
        if random.random() <= self.pc:
            start = random.randint(1, len(parent1) - 1)

        # Perform the actual cross-over
        for i in range(start, len(parent1)):
            child1[i] = parent2[i]
            child2[i] = parent1[i]

        # Find duplicates, if there are any, and replace them
        child1, child2 = self.replace_duplicates(child1, child2)
        return child1, child2

    # Find duplicates, if there are any, and replace them
    # @param child1 - First child chromosome
    # @param child2 - Second child chromosome
    def replace_duplicates(self, child1, child2):
        duplicates1 = []
        duplicates2 = []
        indices1 = []
        indices2 = []

        # Find duplicates in first child
        for i in range(len(child1)):
            if child1.count(child1[i]) > 1 and child1[i] not in duplicates1:
                duplicates1.append(child1[i])
                indices1.append(i)

        if len(duplicates1) == 0:
            return child1, child2

        # Find duplicates in second child
        for i in range(len(child2)):
            if child2.count(child2[i]) > 1 and child2[i] not in duplicates2:
                duplicates2.append(child2[i])
                indices2.append(i)

        for i in range(len(duplicates1)):
            child1[indices1[i]] = duplicates2[i]
            child2[indices2[i]] = duplicates1[i]

        return child1, child2

    # Evaluate the fitness of the whole population and compute the ratios for the selecting and mating process
    # @param population - List of chromosomes representing the population in a generation
    # @param product_to_product - List of distances between the product locations
    # @param start_to_product - List of distances between start coordinates and all product locations
    # @param product_to_end - List of distances between end coordinates and all product locations
    def evaluate(self, population, product_to_product, start_to_product, product_to_end):
        fitness_evaluations = []
        total_fitness = 0
        ratio_list = []

        # Calculate fitness of each chromosome
        for i in range(self.pop_size):
            evaluation = self.fitness(population[i], product_to_product, start_to_product, product_to_end)
            fitness_evaluations.append(evaluation)

        worst_fitness = max(fitness_evaluations)

        # Calculate the difference between each fitness and the worst fitness
        for i in range(self.pop_size):
            fitness_evaluations[i] = worst_fitness - fitness_evaluations[i]
            total_fitness = total_fitness + fitness_evaluations[i]

        # Calculate the ratios for each chromosome, used in the selection process
        for i in range(self.pop_size):
            ratio = fitness_evaluations[i] / total_fitness
            ratio_list.append(ratio)

        return ratio_list

    # Selects 2 parents for the mating process, fitter candidates have a bigger chance of being chosen
    # @param population - List of chromosomes representing the population in a generation
    # @param ratio_list - List of fitness ratios for each chromosome
    def select_parents(self, population, ratio_list):
        random_number = random.random()
        index = 0
        ratio_list_cumulative = []
        ratio_list_cumulative.append(ratio_list[0])

        # Calculate the cumulative ratios
        for i in range(1, len(ratio_list)):
            temp = ratio_list_cumulative[i - 1] + ratio_list[i]
            ratio_list_cumulative.append(temp)

        # Select the first parent
        for i in range(self.pop_size):
            if ratio_list_cumulative[i] > random_number:
                index = i
                break

        parent1 = population[index]

        random_number = random.random()
        index = 0

        # Select the second parent
        for i in range(self.pop_size):
            if ratio_list_cumulative[i] > random_number:
                index = i
                break

        parent2 = population[index]

        return parent1, parent2

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        product_to_product = tsp_data.get_distances()
        start_to_product = tsp_data.get_start_distances()
        product_to_end = tsp_data.get_end_distances()

        # Create the initial population
        population = self.create_initial_population(len(product_to_end))

        # Evaluate the initial population
        ratio_list = self.evaluate(population, product_to_product, start_to_product, product_to_end)

        # Evolve through the generations
        for i in range(self.generations):
            next_generation = []

            # Create the population for the next generation
            while len(next_generation) < self.pop_size:
                parent1, parent2 = self.select_parents(population, ratio_list)
                child1, child2 = self.cross_over(parent1, parent2)
                next_generation.append(self.mutate(child1))
                next_generation.append(self.mutate(child2))

            population = next_generation.copy()

            # Evaluate the new population
            ratio_list = self.evaluate(population, product_to_product, start_to_product, product_to_end)

        # Find the optimal solution in the end
        optimal_solution = population[np.argmax(ratio_list)]


        print(optimal_solution)
        print("route length = ", self.fitness(optimal_solution, product_to_product, start_to_product, product_to_end))

        return optimal_solution