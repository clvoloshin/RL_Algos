
class Snake(object):
    def __init__(self, 
                 action_space,
                 color = [0,0,0], 
                 direction = '0', 
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
        '''
        RGB to Grayscale based on wiki formula
        '''
        R,G,B = self.color
        return 0.299*R + 0.587*G + 0.114*B 

    def grow(self, food):
        '''
        Increases the size of the body by growth 
        Places new body part at the position of the last body part
        '''
        self.length += food.growth
        self.body += [self.previous_tail_location] #TODO, what happens if growth > 1? Currently, no effect

    def move(self):
        '''
        Move snake by adding new head (in appropriate direction) and removing tail
        '''
        new_location = [[self.body[0][0] + self.heading[0], 
                        self.body[0][1] + self.heading[1]]]
        self.body = new_location + self.body
        self.previous_tail_location = self.body.pop()

    def eat(self, all_food):
        '''
        Check if snake's head location is the same as any food location, implying the snake ate food

        Param
            all_food: list
                List of food objects

        Return
            int or None
                Represents which food the snake ate in the all_food list
        '''
        idx_eaten = [False] * len(all_food)
        for idx,food in enumerate(all_food):
            if self.body[0] == food.location[0]:
                self.grow(food)
                return idx
                
        return None

    def set_heading(self, new_direction):
        '''
        Change direction that snake is heading in

        Param
            new_direction: str
                A key to the action_space dictionary holding new (x,y) direction
        '''
        
        # This if statement imposes that a snake not turn around on itself. 
        # Should snake learn this itelf? I think yes.
        #if not self.are_opposites(self, self.heading, self.action_space[new_direction]):
        self.heading = self.action_space[new_direction]

    @staticmethod
    def are_opposites(snake, action, new_action):
        '''
        Check if two actions are negatives of one another

        Param
            snake: object
                an instance of this class

            action: list of length 2
                [x,y] direction

            new_action: list of length 2
                [x,y] direction

        Return
            bool
        '''
        # only matters if snake.length > 2
        return True if ((snake.length > 2) and (action[0] == -new_action[0]) and (action[1] == -new_action[1])) else False



