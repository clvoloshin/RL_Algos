
class Snake():
    def __init__(self, 
                 action_space,
                 color = [0,0,0], 
                 direction = 'up', 
                 start_x = 100, 
                 start_y = 100):
        '''
        By default, initializes a black snake at (0,0) of length 1 (just the head)
        heading up the screen
        '''
        self.action_space = action_space
        self.color = color # (R, G, B)
        self.length = 1
        self.heading = self.action_space[direction]
        self.body = [[start_x,start_y]] 
        self.previous_tail_location = [None,None]
        self.alive = True
        
    def to_grayscale(self):
        R,G,B = self.color
        return 0.299*R + 0.587*G + 0.114*B 

    def grow(self, food):
        '''
        Increases the size of the body by 1
        Places new body part at the position of the last body part
        '''
        self.length += food.growth
        self.body += [self.previous_tail_location]

    def move(self):
        new_location = [[self.body[0][0] + self.heading[0], 
                        self.body[0][1] + self.heading[1]]]
        self.body = new_location + self.body
        self.previous_tail_location = self.body.pop()

    def eat(self, all_food):
        idx_eaten = [False] * len(all_food)
        for idx,food in enumerate(all_food):
            if self.body[0] == food.location[0]:
                self.grow(food)
                return idx
                
        return None

    def set_heading(self, new_direction):
        if not self.are_opposites(self, self.heading, self.action_space[new_direction]):
            self.heading = self.action_space[new_direction]

    @staticmethod
    def are_opposites(snake, action, new_action):
        # only matters if snake.length > 2
        return True if ((snake.length > 2) and (action[0] == -new_action[0]) and (action[1] == -new_action[1])) else False



