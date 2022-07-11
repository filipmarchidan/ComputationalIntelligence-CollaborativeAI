import numpy as np
import matplotlib.pyplot as plt

from main.Maze import Maze
from main.Agent import Agent
from MyQLearning import MyQLearning
from MyEGreedy import MyEGreedy


def get_final_path(learn: MyQLearning, selection: MyEGreedy, maze: Maze, currReward):
    robot = Agent(0, 0)
    curr_state = robot.get_state(maze)

    # Choose best action according to highest q-value:
    while maze.get_reward(curr_state) != currReward:
        action = selection.get_best_action(robot, maze, learn)  # Choose best action
        curr_state = robot.do_action(action, maze)

    return robot.nr_of_actions_since_reset


if __name__ == "__main__":
    # TODO replace this with the location to your maze on your file system
    # load the maze
    file = "..\\..\\data\\toy_maze.txt"
    #file = "..\\..\\data\\easy_maze.txt"
    maze = Maze(file)

    # TODO set target location coordinates according to the maze:
    # Set the reward at the bottom right to 10
    maze.set_reward(maze.get_state(9, 9), 10)     # Toy maze
    maze.set_reward(maze.get_state(9, 0), 5)  # Toy maze
    current_reward = 0
    #maze.set_reward(maze.get_state(24, 14), 10)    # Easy maze

    # TODO Tweak parameters:
    episodes = 100        # No of trials
    no_steps = 30000     # Max steps per trial
    alfa = 0.7
    gamma = 0.9
    epsilon = 0.1
    # -----------------------

    # TODO tweak the number of runs (this is used for plotting):
    numberOfRuns = 10
    avg_steps_per_trial = np.zeros(episodes)

    for x in range(numberOfRuns):
        steps_per_trial = []

        # create a robot at starting and reset location (0,0) (top left)
        robot = Agent(0, 0)

        # make a selection object (you need to implement the methods in this class)
        selection = MyEGreedy()

        # make a Qlearning object (you need to implement the methods in this class)
        learn = MyQLearning()

        stop = False
        trial = 0     # Keeps track of the current trial/episode

        # keep learning until you decide to stop
        while not stop:

            # Reset robot to starting position:
            robot.reset()       # Each reset represents a new trial/episode

            # Robot making steps through the maze:
            for step in range(no_steps):
                # Get (s, a):
                state = robot.get_state(maze)           # Current state s
                action = selection.get_egreedy_action(robot, maze, learn, epsilon)   # Random action according to policy

                # Get (s', r, possible_actions(s')):
                state_next = robot.do_action(action, maze)
                reward = maze.get_reward(state_next)              # Reward at state s'
                possible_actions = maze.get_valid_actions(robot)

                # Update Q according to formula:
                learn.update_q(state, action, reward, state_next, possible_actions, alfa, gamma)

                # ----------------------------
                # If reward of next state is 10, then it's the target - don't do any more steps:
                if reward == 10:
                    current_reward = reward
                    break
                if reward == 5:
                    current_reward = reward
                    break

            # Trial/Episode finished:
            trial += 1
            steps_per_trial.append(robot.nr_of_actions_since_reset)   # Memorize number of steps per trial

            # Stopping criterion:
            if trial >= episodes:
                stop = True
        # ----------------------------------

        # Output nr of steps per trial/episode:
        # for i in range(len(steps_per_trial)):
        #    print('Trial ' + str(i+1) + ' took ' + str(steps_per_trial[i]) + ' steps')

        avg_steps_per_trial += steps_per_trial

        # Output solution:
        best_path = get_final_path(learn, selection, maze, current_reward)
        print("Final path length: " + str(best_path))

        # END OF A RUN OF THE ALGORITHM

    # Average nr of steps per trial:
    avg_steps_per_trial /= numberOfRuns
    #for i in range(len(avg_steps_per_trial)):
    #    print('Trial ' + str(i) + ' took an average of ' + str(avg_steps_per_trial[i]) + ' steps')

    # ----------------------
    # Plot performances:
    plt.plot(np.arange(1, episodes + 1), avg_steps_per_trial, color='orange')
    plt.title("Average no of steps per trial")
    plt.xlabel("Trials")
    plt.ylabel("Avg steps")
    plt.show()
    # ----------------------
