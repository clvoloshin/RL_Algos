import gym
from gym import error, spaces, utils
from gym.utils import seeding
from snake import Snake
from food import Food
import numpy as np
from snake_utils import get_state
from gym.envs.classic_control import rendering
from viewer import newViewer
# from cautious_snake import CautiousSnake
# import pdb

class SnakeEnv(gym.Env):
  #metadata = {'render.modes': ['human']}
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, screen_width = 600, screen_height=400, viewer=None, number_of_snakes = 2):
        self.viewer = viewer
        self.number_of_snakes = number_of_snakes #1 = Classic. >1 is multiplayer
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.min_amount_of_food = np.random.randint(0,5) #...idk what this is supposed to really be
        self.start_number_of_food = np.random.randint(self.min_amount_of_food,self.min_amount_of_food*3)


        self.action_space = {
                            'up'   : [0,1],
                            'down' : [0,-1],
                            'left' : [-1,0],
                            'right': [1,0]
                            }

        self.growth = 1
        
        self.state = None
        self.idxs_of_alive_snakes = np.arange(self.number_of_snakes)
        self.use_grayscale = True
        self.use_raw = False

    def step(self, action_n):
        
        for action in action_n:
            assert self.action_space[action], "%r (%s) invalid"%(action, type(action))
        
        assert len(action_n) == len(self.idxs_of_alive_snakes), 'The number of actions has to equal the number of alive snakes'

        food_eaten = []
        rewards_n = [] 
        count = 0
        for snake in self.snakes:
            if snake.alive:
                snake.set_heading(action_n[count])
                count += 1
                try:
                    snake.move(self.snakes, self.screen_width, self.screen_height)
                except:
                    snake.move()
                which_food = snake.eat(self.food)
                if which_food is not None:
                    food_eaten.append(which_food)
                    rewards_n.append(1)
                else:
                    rewards_n.append(0)
            else:
                rewards_n.append(0)

        self.food = [x for idx,x in enumerate(self.food) if idx not in food_eaten]

        self.state, self.idxs_of_alive_snakes = get_state(self.snakes, self.food, self.screen_width, self.screen_height, self.min_amount_of_food, self.growth)

        done_n = np.array([True]*self.number_of_snakes)
        done_n[self.idxs_of_alive_snakes] = False

        return self.state, rewards_n, done_n

    def reset(self):
        starting_locs = []
        while len(starting_locs) < self.number_of_snakes + self.start_number_of_food: # possible infinite loop in edge case where #_snakes large. Todo: handle this.
            start_x_loc = np.random.randint(1, self.screen_width-1, size = self.number_of_snakes + self.start_number_of_food)
            start_y_loc = np.random.randint(1, self.screen_height-1, size = self.number_of_snakes + self.start_number_of_food)
            starting_locs = list(set(zip(start_x_loc, start_y_loc)))

        self.snakes = [Snake(self.action_space,
                             start_x = starting_locs[idx][0], 
                             start_y = starting_locs[idx][1], 
                             color = np.random.uniform(size=3)) for idx in range(self.number_of_snakes)]

        # self.snakes = [Snake(self.action_space,
        #                      start_x = starting_locs[idx][0], 
        #                      start_y = starting_locs[idx][1], 
        #                      color = np.random.uniform(size=3)) for idx in range(self.number_of_snakes-1, self.number_of_snakes)] + self.snakes

        self.food = [Food(start_x = starting_locs[idx][0],
                          start_y = starting_locs[idx][1],
                          growth = self.growth) for idx in range(self.number_of_snakes, self.number_of_snakes+self.start_number_of_food)]

        self.state, self.idxs_of_alive_snakes = get_state(self.snakes, self.food, self.screen_width, self.screen_height, self.min_amount_of_food, self.growth)

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
    
    def render(self, mode='human', close=False):
        if self.viewer == None:
            self.viewer = newViewer(self.screen_width, self.screen_height)
        #self. viewer.render(return_rgb_array = mode=='rgb_array')

        for idx in self.idxs_of_alive_snakes:
            snake = self.snakes[idx]
            for pixel in snake.body:
                self.viewer.draw_point(pixel, color = (snake.color[0], snake.color[1], snake.color[2]))

        for food in self.food:
            for pixel in food.location:
                self.viewer.draw_point(pixel, color = (food.color[0], food.color[1], food.color[2]))


        return self.viewer.render(return_rgb_array = mode=='rgb_array')




    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]