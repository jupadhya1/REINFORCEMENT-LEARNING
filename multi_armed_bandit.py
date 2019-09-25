#!/usr/bin/env python

# The rewards associated with each arm are modeled by
# Bernoulli distributions return 1 or 0.



#A multiarmed bandit with bernoulli distribution
import numpy as np

class MultiArmedBandit:
    def __init__(self, reward_probability_dist=[0.3, 0.5, 0.8]):
        """ Set the probability for each arms
        """
        self.reward_probability_dist = reward_probability_dist


    def step(self, action):
        """Pull the arm indicated in the 'action' parameter.
        
        """
        if action > len(self.reward_probability_dist):
            raise Exception("MULTI ARMED BANDIT][ERROR] the action" + str(action) + " is out of range, total actions: " + str(len(self.reward_probability_dist)))
        p = self.reward_probability_dist[action]
        q = 1.0-p
        return np.random.choice(2, p=[q, p])


