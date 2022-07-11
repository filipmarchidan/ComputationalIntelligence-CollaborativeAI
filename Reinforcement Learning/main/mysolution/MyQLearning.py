from main.QLearning import QLearning
import numpy as np

class MyQLearning(QLearning):

    """
        Update method like in formula
        q_old is the old estimate of action and state, that we will update
        q_max the best estimate of action from the new possible actions
        r - reward
        gamma - discount factor
        alpha - learning rate
    """
    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        q_old = self.get_q(state, action)

        # Q-values for possible_actions in the next state s':
        qChoices = self.get_action_values(state_next, possible_actions)
        q_max = qChoices[np.argmax(qChoices)]

        """
        Formula for updating q
        """
        q_new = q_old + alfa * (r + (gamma * q_max) - q_old)

        self.set_q(state, action, q_new)
        return
