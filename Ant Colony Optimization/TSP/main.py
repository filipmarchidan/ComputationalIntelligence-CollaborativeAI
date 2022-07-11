import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from TSPData import TSPData
from GeneticAlgorithm import GeneticAlgorithm


if __name__ == "__main__":
    # parameters
    population_size = 600
    generations = 250
    persistFile = "productMatrixDist"

    # setup optimization
    tsp_data = TSPData.read_from_file(persistFile)
    ga = GeneticAlgorithm(generations, population_size)

    # run optimzation and write to file
    solution = ga.solve_tsp(tsp_data)
    tsp_data.write_action_file(solution, "Group_72_actions_TSP.txt")