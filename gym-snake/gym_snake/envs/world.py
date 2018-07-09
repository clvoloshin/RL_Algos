from snake import Snake
from food import Food
from snake_utils import get_state

import numpy as np
import scipy.sparse as sparse
import pdb
from copy import copy, deepcopy

import time

class SnakeWorld(object):
    '''
    Object representing the world holding snake and food.
    '''
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

        self.reward_space = {
                            'eat' : 1, #reward for finding food
                            'die' : -1, #reward for dying
                            'alive' : 0 #reward for staying alive
                            }

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.number_of_snakes = number_of_snakes
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
        '''
        Process dynamics of the SnakeWorld given actions

        Param
            action_n: list
                List of actions that each snake does, action_n[i] = ith snake's action

        Return
            self.state: list 
                Represents new state of SnakeWorld

            self.rewards_n: list
                Reward for each snake

            self.done_n: list
                Whether the snake is dead
        '''


        for action in action_n:
            assert self.action_space[action], "%r (%s) invalid"%(action, type(action))
        
        # Check which snakes ate what food
        food_eaten = []
        self.rewards_n = [] 
        for count,snake in enumerate(self.snakes):
            if snake.alive:
                snake.set_heading(action_n[count])
                snake.move()
                which_food = snake.eat(self.food)
                if which_food is not None:
                    food_eaten.append(which_food)
                    self.rewards_n.append(self.reward_space['eat'])
                else:
                    self.rewards_n.append(self.reward_space['alive'])
            else:
                self.rewards_n.append(self.reward_space['alive'])

        self.food = [x for idx,x in enumerate(self.food) if idx not in food_eaten]

        # Process next state, checking if snakes are alive
        last_done_n = self.done_n
        self.state, self.idxs_of_alive_snakes = get_state(self.snakes, self.food, self.screen_width, self.screen_height, self.growth)

        self.state = [state + self.boundary for state in self.state] # add -1 boundary

        # Figure which snakes are dead
        self.done_n = np.array([True]*self.number_of_snakes)
        self.done_n[self.idxs_of_alive_snakes] = False

        # Replace reward for snakes that just died
        self.rewards_n = np.array(self.rewards_n)

        self.rewards_n[(last_done_n ^ self.done_n)] = self.reward_space['die']

        return self.state, self.rewards_n, self.done_n

    def reset(self):
        '''
        Resets the snake world randomly.

        Return
            self.state: list 
                Represents new state of SnakeWorld
        '''

        # Figure starting locations of snakes and food
        starting_locs = []
        while len(starting_locs) < self.number_of_snakes + self.start_number_of_food: # possible infinite loop in edge case where #_snakes large. Todo: handle this.
            start_x_loc = np.random.randint(1, self.screen_width-1, size = self.number_of_snakes + self.start_number_of_food)
            start_y_loc = np.random.randint(1, self.screen_height-1, size = self.number_of_snakes + self.start_number_of_food)
            starting_locs = list(set(zip(start_x_loc, start_y_loc)))

        
        if self.number_of_snakes == 1:
            # Instantiate snakes with fixed color
            self.snakes = [Snake(self.action_space,
                             start_x = starting_locs[idx][0], 
                             start_y = starting_locs[idx][1], 
                             color = [0,0,1]) for idx in range(self.number_of_snakes)]#np.random.uniform(size=3)) for idx in range(self.number_of_snakes)]
        else:
            #Instatitate snakes with random color
            self.snakes = [Snake(self.action_space,
                             start_x = starting_locs[idx][0], 
                             start_y = starting_locs[idx][1], 
                             color = np.random.uniform(size=3)) for idx in range(self.number_of_snakes)]

        # Instatiate Food
        self.food = [Food(start_x = starting_locs[idx][0],
                          start_y = starting_locs[idx][1],
                          growth = self.growth) for idx in range(self.number_of_snakes, self.number_of_snakes+self.start_number_of_food)]

        # Get Starting state, rewards
        self.state, self.idxs_of_alive_snakes = get_state(self.snakes, self.food, self.screen_width, self.screen_height, self.growth)
        self.state = [state + self.boundary for state in self.state] # add -1 boundary
        self.rewards_n = [] # reset rewards
        self.done_n = np.array([False]*self.number_of_snakes) # set that snakes are not dead

        return self.state 

    def __deepcopy__(self, memo):
        '''
        Custom deepcopy such that no two deepcopies point to same mem location

        Use: new_copy = deepcopy(snkworld). Where snkworld is a pointer to SnakeWorld instance
        '''
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        return result
