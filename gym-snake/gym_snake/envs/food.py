import numpy as np
import scipy.sparse as sparse

class Food(object):
    '''
    Basic object representing food
    '''
    def __init__(self, 
                 start_x = 50, 
                 start_y = 50, 
                 growth=1,
                 color = [0,1,0]):
        self.growth = growth #how much to grow the snake
        self.location = [[start_x, start_y]]
        self.color = color

    def set_growth(self, growth):
        self.growth = growth

    