import numpy as np
import scipy.signal
import pdb

class History(object):
    '''
        Keeps track of the history of observations, actions, and rewards
    '''
    def __init__(self, max_buffer_size):
        self.sequences = []
        self.obs = []
        self.act = []
        self.rew = []
        self.new_obs = []
        self.done = []
        self.lengths = []
        self.timesteps = []
        self.total_timesteps = 0
        self.games_in_buffer = 0
        self.games_played = 0
        self.max_buffer_size = max_buffer_size
        self.buffer_size = 0
        self.thrown_away = 0
        self.total_seen = 0

    def append(self,**kwargs):
        '''
        Adds newest sequence to the history
        
        Requires:
            (1) obs: np.array of observations
            (2) actions: np.array of actions
            (3) rewards: np.array of rewards
        '''
       

        # self.sequences.append(
        #                     {'observations': kwargs['obs'],
        #                     'actions': kwargs['actions'],
        #                     'rewards': kwargs['rewards'],
        #                     'lengths': kwargs['lengths']}
        #                      )
        # self.games_played += 1
        

        observations = kwargs['obs']
        actions = kwargs['actions']
        rewards = kwargs['rewards']
        done = kwargs['done']
        new_observations = kwargs['new_obs']


        self.obs.append(observations)
        self.act.append(actions)
        self.rew.append(rewards)
        self.new_obs.append(new_observations)
        self.done.append(done)

        self.buffer_size += 1
        self.total_seen += 1

        while self.buffer_size > self.max_buffer_size:
                self.obs.pop(0)
                self.act.pop(0)
                self.rew.pop(0)
                self.new_obs.pop(0)
                self.done.pop(0)
                self.buffer_size -= 1
                self.thrown_away += 1

    def sample(self, N, discount):
        idxs = np.array([False] * self.buffer_size)
        idxs[np.random.choice(np.arange(self.buffer_size), N)] = True
    
        obs,acts,rews,new_obs,done = np.vstack(self.obs)[idxs], np.hstack(self.act)[idxs], np.hstack(self.rew)[idxs], np.vstack(self.new_obs)[idxs], np.hstack(self.done)[idxs]

        return np.array([[x.A for x in y] for y in obs]), acts, rews, np.array([[x.A for x in y] for y in new_obs]), done

    def get_last_lengths_of_games(self):
        return np.array(self.lengths[-self.games_played:])

    def get_discounted_rewards(self, discount, use_reward_to_go = False):
        if use_reward_to_go:
            # If reward_to_go then use the future rewards at every step in the path:
            # Q(t) = \sum_{i=t}^T x[i] * gamma^{i-t} = \sum_{i=0}^{T-t} x[t+i] gamma^i
            return np.hstack([self.discount_rewards(path, discount) for path in self.rew])
        else:
            # If not reward_to_go then use the total reward at every step in the path:
            # Q = \sum_{i=0}^T x[i] * gamma^i, not a function of t
            return np.hstack([[self.discount_rewards(path, discount)[0]] * len(path) for path in self.rew])

    def get_total_timesteps(self):
        return self.total_timesteps

    def get_timesteps(self):
        return self.timesteps

    def reset(self):
        self.__init__(self.max_buffer_size)

    def soft_reset(self):
        self.games_played = 0

    def full(self):
        return self.buffer_size == self.max_buffer_size

    @staticmethod
    def discount_rewards(x, gamma): 
        '''
        x: vector of shape (T,1)
        gamma: scalar. Represents the discount
        Returns [\sum_{i=0}^T x[i] * gamma^i, \sum_{i=1}^T x[i] gamma^{i-1}, ...]
        '''
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class SecondHistory(object):
    '''
        Keeps track of the history of observations, actions, and rewards
    '''
    def __init__(self, max_allowable_games_played):
        self.sequences = []
        self.obs = []
        self.act = []
        self.rew = []
        self.pis = []
        self.lengths = []
        self.timesteps = []
        self.total_timesteps = 0
        self.games_in_buffer = 0
        self.games_played = 0
        self.max_allowable_games_played = max_allowable_games_played
        self.buffer_size = 0

    def append(self,**kwargs):
        '''
        Adds newest sequence to the history
        
        Requires:
            (1) obs: np.array of observations
            (2) actions: np.array of actions
            (3) rewards: np.array of rewards
        '''
       

        # self.sequences.append(
        #                     {'observations': kwargs['obs'],
        #                     'actions': kwargs['actions'],
        #                     'rewards': kwargs['rewards'],
        #                     'lengths': kwargs['lengths']}
        #                      )
        # self.games_played += 1
        

        observations = kwargs['obs']
        actions = kwargs['actions']
        rewards = kwargs['rewards']
        lengths = kwargs['lengths']
        pis = kwargs['policies']

        for idx,length in enumerate(lengths):
            self.obs.append(observations[:length,idx,:])
            self.act.append(actions[:length,idx])
            self.rew.append(rewards[:length,idx])
            self.pis.append(pis[:length,idx,:])

            self.lengths.append(length)
            self.games_in_buffer += 1
            self.games_played += 1
            self.buffer_size += length


        self.timesteps.append(max(kwargs['lengths']))
        self.total_timesteps += self.timesteps[-1]

        while self.games_in_buffer > self.max_allowable_games_played:
                self.obs.pop(0)
                self.act.pop(0)
                self.rew.pop(0)
                self.pis.pop(0)
                length = self.lengths.pop(0)
                self.buffer_size -= length
                self.games_in_buffer -= 1

    def sample(self, N, discount):
        idxs = np.array([False] * self.buffer_size)
        idxs[np.random.choice(np.arange(self.buffer_size), N)] = True

        return np.vstack(self.obs)[idxs], np.hstack(self.act)[idxs], np.hstack(self.rew)[idxs], np.vstack(self.pis)[idxs]#self.get_discounted_rewards(discount,True)[idxs]

    def get_last_lengths_of_games(self):
        return np.array(self.lengths[-self.games_played:])

    # def get_all_obs(self):
    #     return np.vstack([path['observations'] for path in self.sequences])

    # def get_all_actions(self):
    #     return np.hstack([path['actions'] for path in self.sequences])

    # def get_all_rewards(self):
    #     return np.hstack([path['rewards'] for path in self.sequences])

    def get_discounted_rewards(self, discount, use_reward_to_go = False):
        if use_reward_to_go:
            # If reward_to_go then use the future rewards at every step in the path:
            # Q(t) = \sum_{i=t}^T x[i] * gamma^{i-t} = \sum_{i=0}^{T-t} x[t+i] gamma^i
            return np.hstack([self.discount_rewards(path, discount) for path in self.rew])
        else:
            # If not reward_to_go then use the total reward at every step in the path:
            # Q = \sum_{i=0}^T x[i] * gamma^i, not a function of t
            return np.hstack([[self.discount_rewards(path, discount)[0]] * len(path) for path in self.rew])

    # def get_total_reward_per_sequence(self):
    #     return [path['rewards'].sum() for path in self.sequences]

    def get_total_timesteps(self):
        return self.total_timesteps

    def get_timesteps(self):
        return self.timesteps

    def reset(self):
        self.__init__(self.max_allowable_games_played)

    def soft_reset(self):
        self.games_played = 0

    @staticmethod
    def discount_rewards(x, gamma): 
        '''
        x: vector of shape (T,1)
        gamma: scalar. Represents the discount
        Returns [\sum_{i=0}^T x[i] * gamma^i, \sum_{i=1}^T x[i] gamma^{i-1}, ...]
        '''
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
