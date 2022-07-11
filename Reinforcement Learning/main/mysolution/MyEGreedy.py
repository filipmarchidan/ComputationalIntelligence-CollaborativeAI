import random
import numpy as np


class MyEGreedy:

    def __init__(self):
        pass

    def get_random_action(self, agent, maze):
        """
            Selects an action at random (uniform distrib.) in State s.
        """
        return random.choice(maze.get_valid_actions(agent))

    def get_best_action(self, agent, maze, q_learning):
        """
            Takes the action with the highest q-value in State s.
        """
        actions = maze.get_valid_actions(agent)    # Get valid actions for agent
        state = agent.get_state(maze)              # Agent's state
        qChoices = q_learning.get_action_values(state, actions)  # Get Q(s,a) for every action in <actions>

        if qChoices.count(qChoices[0]) == len(qChoices):
            return self.get_random_action(agent, maze)   # Choose random action if all q-values are the same
        else:
            argmax = np.argmax(qChoices)
            return actions[argmax]   # Action with the highest value

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        """
            Executes Epsilon-greedy algorithm:
            - with probability (1-eps): take (best) action with highest q-value in State s  (given by get_best_action function)
            - with probability eps:     take a uniform random action                        (given by get_random_action function)

            Encode these options as numbers:
            1 = take best action         (probability 1-eps)
            2 = uniform random action    (probability eps)

            Where epsilon is the exploration parameter.
        """
        random_choice = [1, 2]   # 1 - take best action,  2 - take uniform action
        probabilities = [1 - epsilon, epsilon]

        randomize_action = random.choices(random_choice, probabilities)[0]
        if randomize_action == 1:
            return self.get_best_action(agent, maze, q_learning)     # Best action
        else:
            return self.get_random_action(agent, maze)               # Uniform random action
