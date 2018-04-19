import numpy as np
from copy import copy, deepcopy
import scipy.signal

class Node(object):
    def __init__(self, parent, world, P, n_actors, n_actions, player_number, Ps = None):
        self.parent = parent
        self.children = {}
        self.world = world
        self.N = 0 #visit count
        self.W = 0 #Total action value
        self.Q = 0 #Mean action value
        self.Ps = Ps #prior probabilities of children
        self.P = P  #prior probability for this node

        self.actions = np.arange(n_actions).astype(str)
        self.n_actors = n_actors
        self.id = player_number

        self.action_which_led_to_this_node = None
        self.distance_from_parent = 0

    def select(self, policy):

        values = []
        for key in self.children:
            values.append(policy.get_value(self.children[key]))

        a_t = self.children.keys()[np.argmax(values)] #select next step
        return a_t, self.children[a_t] 

    def sample_others(self, action=None):
        # defines distribution over which other actors sample their actions
        all_actions = np.random.choice(self.actions, self.n_actors) #actions by other snakes
        if action:
            all_actions[self.id] = action

        return all_actions

    def is_leaf(self):

        return self.children == {}

    def expand(self, action):

        for _action in self.actions:
            self.children[_action] = Node(self, None, self.Ps[int(_action)], self.n_actors, len(self.actions), self.id)
            self.children[_action].action_which_led_to_this_node = _action
            self.children[_action].distance_from_parent = self.distance_from_parent + 1

        return self.children[action]

   
    def simulation(self, world, reward, depth = 100, gamma = .9):

        if world.done_n[self.id]:
            return np.array([reward])

        temp_world = deepcopy(world)
        count = 0
        rewards = [reward]
        done = False

        while (count < depth) and not done:
            _, reward_n, done_n = temp_world.step(self.sample_others())
            rewards.append(reward_n[self.id])
            done = done_n[self.id]
            count += 1

        return self.discount_rewards(rewards, gamma)[0]

    @staticmethod
    def discount_rewards(x, gamma): 
        '''
        x: vector of shape (T,1)
        gamma: scalar. Represents the discount
        Returns [\sum_{i=0}^T x[i] * gamma^i, \sum_{i=1}^T x[i] gamma^{i-1}, ...]
        '''
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def has_untried_actions(self):
        return [x for x in self.actions if x not in self.children.keys()]