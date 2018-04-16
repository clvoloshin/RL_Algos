from snake import Snake
from snake_utils import check_which_snakes_are_still_alive
import numpy as np
from snake_utils import get_boards


class CautiousSnake(Snake):
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
        Snake.__init__(self, action_space,
                    start_x = start_x, 
                    start_y = start_y, 
                    color = color)
        self.id = self.to_grayscale() + 10**-12 if self.to_grayscale() == .5 else self.to_grayscale()
        self.prev_previous_tail_location = None

    def __str__(self):
        return str(self.id)

    def get_color(self):
        return self.color

    def move(self, snakes, screen_width, screen_height):
        possible_actions = []
        current_heading = self.heading

        for action in [x for x in self.action_space.keys() if not self.are_opposites(self, current_heading, self.action_space[x])]:
            self.prev_previous_tail_location = self.previous_tail_location
            self.set_heading(action)
            Snake.move(self)
            idxs_of_alive_snakes = [idx for idx,snake in enumerate(snakes) if snake.alive]
            check_which_snakes_are_still_alive(np.array(snakes)[idxs_of_alive_snakes], get_boards(snakes, screen_width, screen_height))
            if self.alive == True:
                possible_actions.append(action)
            else:
                self.alive = True

            self.body += [self.previous_tail_location]
            self.body.pop(0)
            self.previous_tail_location = self.prev_previous_tail_location
            self.heading = current_heading

        if possible_actions == []:
            new_heading = np.random.choice(self.action_space.keys())
            self.set_heading(new_heading)
        else:
            new_heading = np.random.choice(possible_actions)
            self.set_heading(new_heading)

        Snake.move(self)


