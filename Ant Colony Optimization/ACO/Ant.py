import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import random
import numpy as np
from ACO.Route import Route
from ACO.Direction import Direction


# Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = random

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):
        route = Route(self.start)
        steps = []

        while self.current_position != self.end:
            steps.append(self.current_position)

            # Take random step in a direction:
            direction = self.takeStep(steps, route)

            if direction is not None:
                route.add(direction)

        return route

    # Takes one step in a random direction
    # Takes into account the previous steps
    def takeStep(self, steps, route):
        deadEnd = False

        # Positions around the current pos., pheromone values and total surrounding pheromone
        positions, pheromone_values, surroundingPheromone = self.maze.get_surrounding_pheromone(self.current_position)

        # All directions:
        directions = [Direction.east, Direction.north, Direction.west, Direction.south]

        probability = []  # east, north, west, south probabilities

        # If we have nowhere to go we have a dead end:
        if surroundingPheromone == 0:
            deadEnd = True
        else:
            # Go through each position and check if it's valid:
            for i, pos in enumerate(positions):
                # if position was visited, or it is not accessible set pheromone value to 0:
                if pos in steps or not self.maze.in_bounds(pos):
                    pheromone_values[i] = 0

            # Compute all probabilities:
            probability = self.compute_probabilities(positions, pheromone_values)

        dir = None

        # If all probabilities are 0 we have a dead end:
        if not np.any(np.array(probability)):
            deadEnd = True

        # Check if there is a dead end:
        if deadEnd is False:
            # Take a random direction:
            dir = random.choices(directions, probability, k=1)
            # Choose a direction based on the probabilities and check if it is not present in the steps vector:
            while positions[Direction.dir_to_int(dir[0])] == steps[len(steps) - 1] or pheromone_values[Direction.dir_to_int(dir[0])] == 0:
                dir = random.choices(directions, probability, k=1)

            self.current_position = positions[Direction.dir_to_int(dir[0])]
        else:
            # Get last direction followed and opposite:
            last_dir = route.remove_last()[0]
            opposite_dir = self.get_opposite_dir(last_dir)

            # Go back to prev. position:
            last_pos = positions[Direction.dir_to_int(opposite_dir)]
            self.current_position = last_pos

        return dir

    # Compute probabilities in certain positions
    # Formula used from the paper
    def compute_probabilities(self, positions, pheromones):
        alpha = 1.2
        beta = 3

        probabilities = [0, 0, 0, 0]
        s = 0   # sum

        for i, pos in enumerate(positions):
            # Visibility as 1 divided by the distance towards end:
            dist = self.distance_towards_objective(pos, self.end)
            visibility = 1 if dist == 0 else 1/dist
            visibility = visibility ** beta

            trail = pheromones[i] ** alpha

            temp = visibility * trail
            probabilities[i] = temp
            s += temp

        # if sum == 0 we have a dead end:
        if s == 0:
            return probabilities

        for i in range(len(probabilities)):
            probabilities[i] /= s

        return probabilities


    @staticmethod
    def get_opposite_dir(direction: Direction):
        if direction is Direction.north:
            return Direction.south
        elif direction is Direction.south:
            return Direction.north
        elif direction is Direction.east:
            return Direction.west
        elif direction is Direction.west:
            return Direction.east

    @staticmethod
    def distance_towards_objective(position, end):
        return abs(end.get_x() - position.get_x()) + abs(end.get_y() - position.get_y())


