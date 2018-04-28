from snake import Snake
from food import Food
from snake_utils import get_state

import numpy as np
import scipy.sparse as sparse
import pdb
from copy import copy, deepcopy

import time

class SnakeWorld(object):
    def __init__(self, 
                 screen_width,
                 screen_height,
                 number_of_snakes,
                 growth,
                 boundary):

        self.action_space = {
                            '0' : [0,1],
                            '1' : [0,-1],
                            '2' : [-1,0],
                            '3' : [1,0]
                            }

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.number_of_snakes = number_of_snakes
        #self.min_amount_of_food = number_of_snakes #np.random.randint(1,5) #...idk what this is supposed to really be
        self.start_number_of_food = number_of_snakes #np.random.randint(self.min_amount_of_food,self.min_amount_of_food*3)

        self.growth = growth
        self.boundary = boundary

        self.idxs_of_alive_snakes = np.arange(self.number_of_snakes)
        self.use_grayscale = True
        self.use_raw = False

        self.state = []
        self.rewards_n = []
        self.done_n = np.array([False]*self.number_of_snakes)

        

    def step(self, action_n):

        for action in action_n:
            assert self.action_space[action], "%r (%s) invalid"%(action, type(action))
        
        #assert len(action_n) == len(self.idxs_of_alive_snakes), 'The number of actions has to equal the number of alive snakes'

        food_eaten = []
        self.rewards_n = [] 
        for count,snake in enumerate(self.snakes):
            if snake.alive:
                snake.set_heading(action_n[count])
                snake.move()
                which_food = snake.eat(self.food)
                if which_food is not None:
                    food_eaten.append(which_food)
                    self.rewards_n.append(1)
                else:
                    self.rewards_n.append(0)
            else:
                self.rewards_n.append(0)

        self.food = [x for idx,x in enumerate(self.food) if idx not in food_eaten]

        last_done_n = self.done_n
        self.state, self.idxs_of_alive_snakes = get_state(self.snakes, self.food, self.screen_width, self.screen_height, self.growth)

        self.state = [state + self.boundary for state in self.state] # add -1 boundary

        self.done_n = np.array([True]*self.number_of_snakes)
        self.done_n[self.idxs_of_alive_snakes] = False

        #snakes that just died
        self.rewards_n = np.array(self.rewards_n)
        self.rewards_n[(last_done_n ^ self.done_n)] = -1

        return self.state, self.rewards_n, self.done_n

    def reset(self):
        starting_locs = []
        while len(starting_locs) < self.number_of_snakes + self.start_number_of_food: # possible infinite loop in edge case where #_snakes large. Todo: handle this.
            start_x_loc = np.random.randint(1, self.screen_width-1, size = self.number_of_snakes + self.start_number_of_food)
            start_y_loc = np.random.randint(1, self.screen_height-1, size = self.number_of_snakes + self.start_number_of_food)
            starting_locs = list(set(zip(start_x_loc, start_y_loc)))

        self.snakes = [Snake(self.action_space,
                             start_x = starting_locs[idx][0], 
                             start_y = starting_locs[idx][1], 
                             color = [0,0,1]) for idx in range(self.number_of_snakes)]#np.random.uniform(size=3)) for idx in range(self.number_of_snakes)]

        # self.snakes = [Snake(self.action_space,
        #                      start_x = starting_locs[idx][0], 
        #                      start_y = starting_locs[idx][1], 
        #                      color = np.random.uniform(size=3)) for idx in range(self.number_of_snakes-1, self.number_of_snakes)] + self.snakes

        self.food = [Food(start_x = starting_locs[idx][0],
                          start_y = starting_locs[idx][1],
                          growth = self.growth) for idx in range(self.number_of_snakes, self.number_of_snakes+self.start_number_of_food)]

        self.state, self.idxs_of_alive_snakes = get_state(self.snakes, self.food, self.screen_width, self.screen_height, self.growth)
        self.state = [state + self.boundary for state in self.state] # add -1 boundary
        self.rewards_n = []
        self.done_n = np.array([False]*self.number_of_snakes)
        # states_n = []
        # if not self.use_raw:
        #     for i in self.idxs_of_alive_snakes:
        #         snake_i_copy_of_state = self.state.copy()            
        #         for j in self.idxs_of_alive_snakes:
        #             body = np.array(self.snakes[j].body)
        #             if i != j:
        #                 snake_i_copy_of_state[body[:,0],body[:,1]] = 1. # Enemy are 1 <- arbitrary. Assumes grayscale
        #             else:
        #                 snake_i_copy_of_state[body[:,0],body[:,1]] = .25 # You are .25 <- arbitrary. Assumes grayscale
        #         states_n.append(snake_i_copy_of_state)
        # else:
        #     for i in self.idxs_of_alive_snakes:
        #         states_n.append(self.state.copy())

        return self.state 

    # def __new__(cls, *args, **kwargs):
    #     print "Creating Instance"
    #     instance = SnakeWorld.__new__(cls, *args, **kwargs)
    #     return instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        return result
