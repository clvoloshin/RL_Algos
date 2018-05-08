import numpy as np
import scipy.signal
import pdb

class History(object):
    '''
        Keeps track of the history of observations, actions, and rewards
    '''
    def __init__(self, max_buffer_size):
        self.data = []
        # self.obs = []
        # self.act = []
        # self.rew = []
        # self.new_obs = []
        # self.done = []
        self.next_idx = 0
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

        observations = kwargs['obs']
        actions = kwargs['actions']
        rewards = kwargs['rewards']
        done = kwargs['done']
        new_observations = kwargs['new_obs']

        data = (observations, actions, rewards, new_observations, done)

        if self.next_idx >= self.buffer_size:
            self.data.append(data)
            self.buffer_size += 1
        else: 
            self.data[self.next_idx] = data
            self.thrown_away += 1

        self.next_idx = (self.next_idx + 1) % self.max_buffer_size

    def unpack(self, idxs):
        data = np.array(self.data)[idxs]
        obs, act, rew, new_obs, done = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]

        obs, acts, rews, new_obs, done = np.vstack(obs), np.hstack(act), np.hstack(rew), np.vstack(new_obs), np.hstack(done)
        return obs, acts, rews, new_obs, done

    def sample(self, N, discount, is_sparse=True):
        idxs = np.array([False] * self.buffer_size)
        idxs[np.random.choice(np.arange(self.buffer_size), N, replace=False)] = True

        obs, acts, rews, new_obs, done = self.unpack(idxs)
        
        if is_sparse:
            return np.array([[x.A for x in y] for y in obs]), acts, rews, np.array([[x.A for x in y] for y in new_obs]), done
        else:
            return obs, acts, rews, new_obs, done

    # def get_last_lengths_of_games(self):
    #     return np.array(self.lengths[-self.games_played:])

    # def get_discounted_rewards(self, discount, use_reward_to_go = False):
    #     if use_reward_to_go:
    #         # If reward_to_go then use the future rewards at every step in the path:
    #         # Q(t) = \sum_{i=t}^T x[i] * gamma^{i-t} = \sum_{i=0}^{T-t} x[t+i] gamma^i
    #         return np.hstack([self.discount_rewards(path, discount) for path in self.rew])
    #     else:
    #         # If not reward_to_go then use the total reward at every step in the path:
    #         # Q = \sum_{i=0}^T x[i] * gamma^i, not a function of t
    #         return np.hstack([[self.discount_rewards(path, discount)[0]] * len(path) for path in self.rew])

    def get_total_timesteps(self):
        return self.total_timesteps

    def get_timesteps(self):
        return self.timesteps

    def reset(self):
        self.__init__(self.max_buffer_size)

    def soft_reset(self):
        self.games_played = 0

    def full(self, N=None):
        if N:
            return self.buffer_size > min(N, self.max_buffer_size)
        else:
            return self.buffer_size == self.max_buffer_size

    @staticmethod
    def discount_rewards(x, gamma): 
        '''
        x: vector of shape (T,1)
        gamma: scalar. Represents the discount
        Returns [\sum_{i=0}^T x[i] * gamma^i, \sum_{i=1}^T x[i] gamma^{i-1}, ...]
        '''
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class PrioritizedHistory(History):
    '''
        Keeps track of the history of observations, actions, and rewards
    '''
    def __init__(self, max_allowable_games_played, alpha, eps=1e-8):
        super(PrioritizedHistory, self).__init__(max_allowable_games_played)
        
        assert (alpha >= 0) and (alpha <= 1)# If = 0 then uniform, if =1 then full priority
        self.alpha = alpha
        self.max_priority = 1.
        self.priorities = []
        self.priority_eps = eps

    def append(self, *args, **kwargs):
        idx = self.next_idx
        super(PrioritizedHistory, self).append(*args, **kwargs)
        try:
            self.priorities[idx] = self.max_priority ** self.alpha
        except:
            self.priorities.append(self.max_priority ** self.alpha)

    def sample_proportional(self, N):
        # More efficient w tree if buffer_size is large, as in paper/openAi implementation
        return np.random.choice(np.arange(self.buffer_size), size=N, p=np.array(self.priorities)/sum(self.priorities))

    def sample(self, N, is_sparse=True, beta=1.):
        idxs = self.sample_proportional(N)
        s = sum(self.priorities)


        weights = []
        p_min = min(self.priorities) / s
        max_weight = (p_min * self.buffer_size) ** (-beta)

        for idx in idxs:
            p_sample = self.priorities[idx] / s
            weight = (p_sample * self.buffer_size) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        
        obs, acts, rews, new_obs, done = self.unpack(idxs)

        if is_sparse:
            return np.array([[x.A for x in y] for y in obs]), acts, rews, np.array([[x.A for x in y] for y in new_obs]), done, weights, idxs
        else:
            return obs, acts, rews, new_obs, done, weights, idxs

    def update_priorities(self, idxs, priorities):
        assert len(idxs) == len(priorities)
        for idx, priority in zip(idxs, priorities):
            assert priority > 0
            assert 0 <= idx < self.buffer_size
            self.priorities[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)


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

class Reservoir(History):
    def __init__(self, delta, max_buffer_size):
        '''
        Reservoir Buffer used for Reservoir Sampling

        Param
            delta: float >= 1 (beta in the paper. Not to be confused w Prioritized buffer beta)
                If = 1 then uniform reservoir sampling
                If > 1 then exponential reservoir sampling

        '''                
        super(Reservoir, self).__init__(max_buffer_size)
        assert delta >= 1
        self.delta = delta
    
    def sample_proportional(self, k):
        idxs = range(k)

        number_seen = k
        for j in range(k+1, self.buffer_size):
            if np.random.uniform() < (k/(self.delta * number_seen)):
                idx_to_replace = np.random.choice(range(k), 1)
                idxs[idx_to_replace[0]] = j

            number_seen += 1

        return np.array(idxs)

    def unpack(self, idxs):
        data = np.array(self.data)[idxs]
        obs, act = data[:,0],data[:,1]

        obs, acts = np.vstack(obs), np.hstack(act)
        return obs, acts

    def sample(self, k, is_sparse=True):
        '''
            p_k ~= k/(beta*buffer_size) (assuming e^(-x) = 1-x for x small)
        '''

        k = min(k, self.buffer_size)
        idxs = self.sample_proportional(k)
        
        if idxs.shape[0]:
            obs, acts = self.unpack(idxs)
        else:
            return None, None

        if is_sparse:
            return np.array([[x.A for x in y] for y in obs]), acts
        else:
            return obs, acts

    def append(self,**kwargs):
        

        observations = kwargs['obs']
        actions = kwargs['actions']
        
        data = (observations, actions)
        self.data.append(data)
        self.buffer_size += 1

        if self.buffer_size == self.max_buffer_size:
            self.data.pop(0)
            self.buffer_size -= 1
            







