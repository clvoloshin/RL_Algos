import numpy as np
import pdb
from copy import copy, deepcopy
from train import get_data
import pdb

class UCB(object):
    """
    The upper confidence bound
    """
    def __init__(self, c):
        self.c = c

    def get_value(self, node):
        # Alpha zero PUCT algorithm
        return node.Q + self.c * node.P * np.sqrt(node.parent.N) / (1 + node.N)

def find_leaf(node, tree_policy, raw_obs = [], world=None, a_t = None, rew = None):
    
    if node.is_leaf(): #small action space, breadth first
        if a_t is None:
            a_t = np.random.choice(node.actions)
            node = node.expand(a_t) 

            all_actions = node.sample_others(a_t)
            obs,rew,_ = world.step(all_actions)
            raw_obs.append(np.array(obs))
            raw_obs = raw_obs[-2:]
        else:
            assert rew is not None, 'a_t and rew must come together'

        return node, world, np.array(rew)[node.id], raw_obs
    else:
        a_t, node = node.select(tree_policy)
        
        all_actions = node.sample_others(a_t)
        obs,rew,_ = world.step(all_actions)
        is_dead = world.done_n[node.id]

        raw_obs.append(np.array(obs))
        raw_obs = raw_obs[-2:]
        
        if is_dead:
            return node, world, np.array(rew)[node.id], raw_obs
        else:
            return find_leaf(node, tree_policy, raw_obs, world, a_t, rew)

def backup(node, q):
    """
    backup
    """

    while node is not None:

        node.N += 1 #visit count
        node.W += q/(node.n_actors * len(node.actions)) #Expected total action value
        node.Q = node.W/node.N #Mean action value

        node = node.parent


class MCTS(object):
    def __init__(self, tree_policy, backup, action_space_dim, n_actors):
        self.tree_policy = tree_policy
        self.backup = backup
        self.action_space_dim = action_space_dim
        self.n_actors = n_actors

    def run(self, sess, network, world, root, iterations = 1600, tau = 1):

        if root.parent is not None:
            raise ValueError("Root's parent must be None.")
        
        for i in range(iterations):
            
            node, _, rew, raw_obs = find_leaf(root, self.tree_policy, world = deepcopy(world))
            
            # Expectimax-ish
            #reward = node.simulation(world, rew)
            ob = get_data(np.array(raw_obs)[-2:])[node.id:node.id+1]
            P, v = sess.run([network.softmax_policy, network.pred_Q], feed_dict={network.obs: np.array([[y.A for y in x] for x in ob]), 
                                                            network.training_flag: False})

            node.Ps = P[0]

            self.backup(node, v[0][0])

        pi = [0] * len(root.actions)
        for action in np.arange(len(root.actions)): 
            pi[action] = root.children[str(action)].N ** (1/tau)
        
        pi = np.array(pi)
        return pi / float(np.sum(pi))


