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

    def __init__(self, screen_width = 6, screen_height=6, viewer=None, n_actors = 1):
        self.viewer = viewer
        self.n_actors = n_actors #1 = Classic. >1 is multiplayer
        
        self.screen_width = screen_width +2 # +2 for boundary
        self.screen_height = screen_height +2 # +2 for boundary
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

        # represents the -1 boundary around the snake grid
        self.bound_rows, self.bound_cols = rows,cols
        self.boundary = sparse.coo_matrix(([-1]*len(self.bound_rows), (self.bound_rows,self.bound_cols)),  shape = (self.screen_width, self.screen_height)).tocsr()

        self.state = None
        self.world = SnakeWorld(self.screen_width,
                                self.screen_height,
                                self.n_actors, 
                                self.growth,
                                self.boundary)

    def step(self, action_n):
        '''
        Get the state, reward, dones from the world

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
        return self.world.step(action_n) 

    def reset(self):
        '''
        Resets the snake world.

        Return
            self.state: list 
                Represents new state of SnakeWorld
        '''
        return self.world.reset()               
    
    def render(self, mode='human', close=False, headless= False):
        '''
        Renders the snake world
        
        Param
            mode: str
                if 'rgb_array', return will be viewer and array 

            close: bool
                ? TODO

            headless: bool
                Changes return type from viewer+array to just array
                Allows you to run this function even on display-less server. 
                (Yes, this can be done many ways. I found this way to be very easy and intuitive) 

        Return
            rgb_array: numpy array
                Represents gridworld in RGB [0-1., 0-1., 0-1.]

            and possibly:

            viewer: Viewer
                see gym.envs.classic_control.rendering
            


        '''
        if not headless:
            if self.viewer == None:
                from gym.envs.classic_control import rendering
                from viewer import newViewer
                self.viewer = newViewer(self.screen_width, self.screen_height)
            #self. viewer.render(return_rgb_array = mode=='rgb_array')


            for idx in self.world.idxs_of_alive_snakes:
                snake = self.world.snakes[idx]
                for x, pixel in enumerate(snake.body):
                    if x:
                        self.viewer.draw_point(pixel, color = (snake.color[0], snake.color[1], snake.color[2]))
                    else:
                        self.viewer.draw_point(pixel, color = (snake.color[0], snake.color[1], snake.color[2]/2.)) # Head should be slightly different in color

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
                for x, pixel in enumerate(snake.body):
                    if x:
                        rgb_array[self.screen_height - pixel[1] -1, pixel[0] ,:] = snake.color # Assume (0,0) at bottom left for printing. 
                                                                                           # Artifact from the way the gifs are created later
                                                                                           # Has to be done for the gif to match up with the actual array of the board
                    else:
                        rgb_array[self.screen_height - pixel[1] -1, pixel[0] ,:] = [snake.color[0], snake.color[1], snake.color[2]/2.] # Head should be slightly different in color
            for food in self.world.food:
                for pixel in food.location:
                    rgb_array[self.screen_height - pixel[1] -1, pixel[0],:] = food.color

            for pixel in zip(self.bound_rows, self.bound_cols):
                    rgb_array[self.screen_height - pixel[1] -1, pixel[0],:] = [1,0,0]

            return rgb_array