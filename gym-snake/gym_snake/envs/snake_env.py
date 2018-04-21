import gym
from gym import error, spaces, utils
from gym.utils import seeding
from world import SnakeWorld
import numpy as np
import scipy.sparse as sparse
import pdb
# from cautious_snake import CautiousSnake



class SnakeEnv(gym.Env):
  #metadata = {'render.modes': ['human']}
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, screen_width = 8, screen_height=12, viewer=None, n_actors = 1):
        self.viewer = viewer
        self.n_actors = n_actors #1 = Classic. >1 is multiplayer
        
        self.screen_width = screen_width +2 # +2 for boundary
        self.screen_height = screen_height +2 # +2 for boundary
        self.min_amount_of_food = np.random.randint(0,5) #...idk what this is supposed to really be
        self.start_number_of_food = np.random.randint(self.min_amount_of_food,self.min_amount_of_food*3)
        self.growth = 1

        #left and right border
        rows = [0] * (self.screen_height-2) + [self.screen_width-1] * (self.screen_height-2)  
        cols = [x for x in range(1,self.screen_height-1)]*2 

        #up and down border
        rows += [x for x in range(1,self.screen_width-1)]*2 
        cols += [0] * (self.screen_width-2) + [self.screen_height-1] * (self.screen_width-2)  

        #corners
        rows += [0]*2 + [self.screen_width-1]*2
        cols += [0] + [self.screen_height-1] + [0] + [self.screen_height-1]

        self.bound_rows, self.bound_cols = rows,cols
        self.boundary = sparse.coo_matrix(([-1]*len(self.bound_rows), (self.bound_rows,self.bound_cols)),  shape = (self.screen_width, self.screen_height)).tocsr()

        self.state = None
        self.world = SnakeWorld(self.screen_width,
                                self.screen_height,
                                self.n_actors, 
                                self.min_amount_of_food, 
                                self.start_number_of_food,
                                self.growth,
                                self.boundary)

    def step(self, action_n):
        
        return self.world.step(action_n) 

    def reset(self):

        return self.world.reset()               
    
    def render(self, mode='human', close=False, headless= False):
        
        if not headless:
            if self.viewer == None:
                from gym.envs.classic_control import rendering
                from viewer import newViewer
                self.viewer = newViewer(self.screen_width, self.screen_height)
            #self. viewer.render(return_rgb_array = mode=='rgb_array')


            for idx in self.world.idxs_of_alive_snakes:
                snake = self.world.snakes[idx]
                for pixel in snake.body:
                    self.viewer.draw_point(pixel, color = (snake.color[0], snake.color[1], snake.color[2]))

            for food in self.world.food:
                for pixel in food.location:
                    self.viewer.draw_point(pixel, color = (food.color[0], food.color[1], food.color[2]))

            for pixel in zip(self.bound_rows, self.bound_cols):
                    self.viewer.draw_point(pixel, color = (1,0,0))

            return self.viewer.render(return_rgb_array = mode=='rgb_array')

        else:

            rgb_array = np.ones((self.screen_height, self.screen_width, 3))

            for idx in self.world.idxs_of_alive_snakes:
                snake = self.world.snakes[idx]
                for pixel in snake.body:
                    rgb_array[self.screen_height - pixel[1] -1, pixel[0] ,:] = snake.color # Assume (0,0) at bottom left for printing. 
                                                                                           # Artifact from the way the gifs are created later
                                                                                           # Has to be done for the gif to match up with the actual array of the board
            for food in self.world.food:
                for pixel in food.location:
                    rgb_array[self.screen_height - pixel[1] -1, pixel[0],:] = food.color

            for pixel in zip(self.bound_rows, self.bound_cols):
                    rgb_array[self.screen_height - pixel[1] -1, pixel[0],:] = [1,0,0]

            return rgb_array


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]