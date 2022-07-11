import time
from AntColonyOptimization import AntColonyOptimization
from Maze import Maze
from PathSpecification import PathSpecification


def run_algorithm_for_params(ants_gen, no_gen, q, evap, maze_file="easy"):
    # construct the optimization objects
    maze = Maze.create_maze("./../data/" + maze_file + " maze.txt")
    spec = PathSpecification.read_coordinates("./../data/" + maze_file + " coordinates.txt")
    aco = AntColonyOptimization(maze, gen, no_gen, q, evap)

    # save starting time
    start_time = int(round(time.time() * 1000))

    # run optimization
    shortest_route, route_len_generations = aco.find_shortest_route(spec)

    time_taken = (int(round(time.time() * 1000)) - start_time) / 1000.0

    return time_taken, shortest_route


# Driver function for Assignment 1
if __name__ == '__main__':
    # parameters
    gen = 16  # ants per gen
    no_gen = 400  # total no of gen
    q = 100
    evap = 0.7

    times = []
    routes = []
    ants = [5, 10, 14, 15, 16, 20, 25]
    qs = [0.5, 0.6, 0.7, 0.8, 0.9]
    no_gens = [10, 50, 100, 200, 300, 400]
    # TODO: vary parameters and plot route_lengths. Durations can be plotted as a separate plot.
    # time over generations:
    # shortest route in each generation:
    for i in range(len(ants)):
        # gen_current = no_gens[i]
        timeFinal, route_lengths = run_algorithm_for_params(ants_gen=16, no_gen=400, q=q, evap=0.5,
                                                            maze_file="easy")
        routes.append(route_lengths.size())
        times.append(timeFinal)

    print(times)
    print(routes)
    print(no_gens)
