import numpy as np
import sys

ACTION_UP    =  0
ACTION_RIGHT =  1
ACTION_DOWN  =  2
ACTION_LEFT  =  3



class GridWorld(object):
    def __init__(self,shape=[4,4]):
        self.shape = shape


        numStates  = shape[0] * shape[1]
        numActions = 4
        self.numStates = numStates
        self.numActions = numActions
        
        xmax = shape[0]
        ymax = shape[1]

        grid = np.arange(numStates).reshape(shape)

        Model = {}

        x_indices = np.arange(xmax)
        y_indices = np.arange(ymax)

        for x in x_indices:
            for y in y_indices:
                state = y + x*(xmax)
                #print(x,y,state)
                Model[state] ={action:[] for action in np.arange(numActions)}

                is_terminal_state = lambda state : state == 0 or state == (numStates-1)
                reward = 0.0 if is_terminal_state(state) else -1.0


                if is_terminal_state(state):
                    Model[state][ACTION_UP] = [(1.0,state,reward,True)]
                    Model[state][ACTION_RIGHT] = [(1.0,state,reward,True)]
                    Model[state][ACTION_DOWN] = [(1.0,state,reward,True)]
                    Model[state][ACTION_LEFT] = [(1.0,state,reward,True)]
                else:
                    next_state = {}
                    next_state[ACTION_UP] = state if x == 0 else state - ymax
                    next_state[ACTION_RIGHT] = state if y == ymax-1 else state +1
                    next_state[ACTION_DOWN] = state if x == xmax-1 else state + ymax
                    next_state[ACTION_LEFT] = state if y == 0 else state -1 
                    Model[state][ACTION_UP] = [(1.0,next_state[ACTION_UP] ,reward,is_terminal_state(next_state[ACTION_UP]))]
                    Model[state][ACTION_RIGHT] = [(1.0,next_state[ACTION_RIGHT],reward,is_terminal_state(next_state[ACTION_RIGHT]))]
                    Model[state][ACTION_DOWN] = [(1.0,next_state[ACTION_DOWN],reward,is_terminal_state(next_state[ACTION_DOWN]))]
                    Model[state][ACTION_LEFT] = [(1.0,next_state[ACTION_LEFT],reward,is_terminal_state(next_state[ACTION_LEFT]))]
        self.model = Model
            


