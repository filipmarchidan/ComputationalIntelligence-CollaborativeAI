import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import time
from ACO.Maze import Maze
from ACO.PathSpecification import PathSpecification
from ACO.Ant import Ant
from ACO.Route import Route

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

    # Loop that starts the shortest path process
    # @param spec Specification of the route we wish to optimize
    # @return ACO optimized route
    def find_shortest_route(self, path_specification, get_generations_routes=False):
        self.maze.reset()       # reset pheromone qty

        # Keep track of shortest routes of each generation
        shortest_routes_list = []
        shortest_route = None

        # Loop through generations:
        for t in range(self.generations):
            routes = []   # routes of one generation

            for i in range(self.ants_per_gen):    # i - ant

                # Create ant and find a route
                ant = Ant(self.maze, path_specification)
                route = ant.find_route()

                # Case when shortest route is none:
                if shortest_route is None:
                    shortest_route = route

                # If route is shorter memorize it
                if route.shorter_than(shortest_route):
                    shortest_route = route

                # Append route to list
                routes.append(route)

            # After one generation adjust pheromone:
            self.maze.evaporate(self.evaporation)            # evaporate pheromone
            self.maze.add_pheromone_routes(routes, self.q)   # add pheromone

            # Print progress
            shortest_route_len = shortest_route.size()
            self.print_progress(t, self.generations, shortest_route_len)

            # Add shortest route found in this generation to list
            if get_generations_routes:
                shortest_routes_list.append(shortest_route.size())

        return shortest_route, shortest_routes_list

    @staticmethod
    def print_route(route: Route):
        c = route.get_start()
        r = route.get_route()

        print("Taking the following route:")

        for i in range(len(r)):
            dir = r[i]            # for some reason dir is a list of one element
            c = c.add_direction(dir[0])
            print(c)

    @staticmethod
    def print_progress(x, total_gen, route_length):
        print("Generation " + str(x) + "/" + str(total_gen) + " ready.   Best route length:  " + str(route_length), end='\r')
