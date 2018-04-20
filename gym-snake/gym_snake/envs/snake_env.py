import gym
from gym import error, spaces, utils
from gym.utils import seeding
from world import SnakeWorld
import numpy as np
# from cautious_snake import CautiousSnake



class SnakeEnv(gym.Env):
  #metadata = {'render.modes': ['human']}
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, screen_width = 20, screen_height=20, viewer=None, n_actors = 1):
        self.viewer = viewer
        self.n_actors = n_actors #1 = Classic. >1 is multiplayer
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.min_amount_of_food = np.random.randint(0,5) #...idk what this is supposed to really be
        self.start_number_of_food = np.random.randint(self.min_amount_of_food,self.min_amount_of_food*3)
        self.growth = 1
        
        self.state = None
        self.world = SnakeWorld(self.screen_width,
                                self.screen_height,
                                self.n_actors, 
                                self.min_amount_of_food, 
                                self.start_number_of_food,
                                self.growth)

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

            return rgb_array


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]