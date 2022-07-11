import os, sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import traceback
from ACO.Coordinate import Coordinate
from ACO.Direction import Direction

# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:

    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length):
        self.walls = walls
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.pheromones = None
        self.initialize_pheromones()

    # Initialize pheromones to a start value.
    def initialize_pheromones(self):
        self.pheromones = np.ones(shape=(self.width, self.length))

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # Update the pheromones along a certain route according to a certain Q
    # @param r The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, route, q):
        r: list = route.get_route()            # route - list of directions
        c: Coordinate = route.get_start()      # start coordinate
        route_len = route.size()

        # Coordinates are x and y, not col. and row!
        if route_len !=0 :
            self.pheromones[c.get_x(), c.get_y()] += q / route_len   # add pheromone to start coordinate

        # Go through the route and adjust pheromone quantity:
        for i in range(len(r)):
            dir = r[i]            # for some reason dir is a list of one element
            c = c.add_direction(dir[0])
            self.pheromones[c.get_x(), c.get_y()] += q / route_len   # q normalized by route length


    # Update pheromones for a list of routes
    # @param routes A list of routes
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, routes, q):
        for r in routes:
            self.add_pheromone_route(r, q)

    # Evaporate pheromone
    # @param rho evaporation factor
    def evaporate(self, rho):
        self.pheromones *= rho

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Returns the amount of pheromones on the neighbouring positions (N/S/E/W).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, position: Coordinate):
        # positions of each coordinate:
        pos_E = position.add_direction(Direction.east)
        pos_N = position.add_direction(Direction.north)
        pos_W = position.add_direction(Direction.west)
        pos_S = position.add_direction(Direction.south)

        positions = [pos_E, pos_N, pos_W, pos_S]

        E = self.get_pheromone(pos_E)
        N = self.get_pheromone(pos_N)
        W = self.get_pheromone(pos_W)
        S = self.get_pheromone(pos_S)

        pheromone = [E, N, W, S]
        s = N + S + E + W
        return positions, pheromone, s

    # Pheromone getter for a specific position. If the position is not in bounds returns 0
    # @param pos Position coordinate
    # @return pheromone at point
    def get_pheromone(self, pos: Coordinate):
        if self.in_bounds(pos):
            return self.pheromones[pos.get_x()][pos.get_y()]
        else:
            return 0

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, position):
        return position.x_between(0, self.width) and position.y_between(0, self.length) and self.walls[position.get_x()][position.get_y()] == 1

    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a mze from a file
    # @param filePath Path to the file
    # @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])

            #make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])

            for y in range(length):
                line = lines[y+1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()

    def print_pheromones(self, decimals=1):
        print('\n'.join([''.join(['{:8}'.format(round(item, decimals)) for item in row])
                         for row in self.pheromones.T]))