import time
from AntColonyOptimization import AntColonyOptimization
from Maze import Maze
from PathSpecification import PathSpecification

# Runs the algorithm for a specific maze file
def run_algorithm_for_params(ants_gen, no_gen, q, evap, maze_file="easy"):
    # construct the optimization objects
    maze = Maze.create_maze("./../data/" + maze_file + " maze.txt")
    spec = PathSpecification.read_coordinates("./../data/" + maze_file + " coordinates.txt")
    aco = AntColonyOptimization(maze, gen, no_gen, q, evap)

    # save starting time
    start_time = int(round(time.time() * 1000))

    # run optimization
    shortest_route, routes_list_over_generations = aco.find_shortest_route(spec)

    # ---------
    # Print shortest route and pheromone amounts:
    # aco.print_route(shortest_route)
    # maze.print_pheromones(decimals=1)
    # ---------

    # print time taken
    print("Time taken for maze " + maze_file + ":" + str((int(round(time.time() * 1000)) - start_time) / 1000.0))

    # save solution
    shortest_route.write_to_file("Group_72_" + maze_file + ".txt")

    # print route size
    print("Route size for maze " + maze_file + ":" + str(shortest_route.size()))

# Driver function for Assignment 1
if __name__ == '__main__':
    # parameters
    gen = 16
    no_gen = 200
    q = 100
    evap = 0.5

    run_algorithm_for_params(gen, no_gen, q, evap, maze_file="easy")

    # parameters
    gen = 16
    no_gen = 450
    q = 100
    evap = 0.5

    run_algorithm_for_params(gen, no_gen, q, evap, maze_file="medium")

    # parameters
    gen = 16
    no_gen = 500
    q = 100
    evap = 0.5

    run_algorithm_for_params(gen, no_gen, q, evap, maze_file="hard")