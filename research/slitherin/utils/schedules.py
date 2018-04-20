
class LinearSchedule(object):
    def __init__(self, num_steps=1000, initial_value=1.0, final_value=0):
        self.num_steps = num_steps
        self.initial_value = initial_value
        self.final_value = final_value
        self.steps_taken = None

    def value(self, timestep):
        t = min(float(timestep) / self.num_steps, 1.0)
        return t * (self.final_value) + (1-t) * (self.initial_value)

    def next(self):
        if self.steps_taken is not None:
            self.steps_taken += 1
        else:
            self.steps_taken = 0
        
        return self.value(self.steps_taken)

    def reset():
        self.steps_taken = None
