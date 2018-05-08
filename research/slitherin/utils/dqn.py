import tensorflow as tf
import numpy as np
import pdb
from history import History, PrioritizedHistory, Reservoir

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

class DQN(object):
    def __init__(self, 
                 sess,
                 model,
                 state_shape, 
                 writer,
                 n_actions=4, 
                 batch_size=32,
                 gamma=.99,
                 update_freq=1000,
                 ddqn=True, # double dqn
                 buffer_size = 10000,
                 clip_grad = None,
                 batches_per_epoch = 1,
                 is_sparse = True,
                 use_priority = False,
                 priority_alpha = .5,
                 priority_eps = 1e-8,
                 _id = '0'
                 ):

        self.sess = sess
        self.model = model
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_freq = update_freq
        self.ddqn = ddqn
        self.buffer_size = buffer_size
        self.clip_grad = clip_grad
        self.is_sparse = is_sparse

        self.momentum = .9
        self.weight_decay = .9

        if use_priority:
            self.buffer = PrioritizedHistory(self.buffer_size, priority_alpha, priority_eps)
        else:
            self.buffer = History(self.buffer_size)

        self.epoch = 0
        self.batches_per_epoch = batches_per_epoch

        self.summary_writer = writer

        self.loss_func = huber_loss

        self.id = str(_id)
        self.scope = 'main' + self.id
        self.reuse = None
        
        self.build()

    def __str__(self):
        return 'DQN %s' % self.id


    @staticmethod
    def copy_to_target(source_, target_):
        theta_set = []
        for source_var, target_var in zip(sorted(source_, key=lambda v: v.name), 
                                            sorted(target_, key=lambda v: v.name)):
            theta_set.append(target_var.assign(source_var)) # target <- source
        return tf.group(*theta_set)

    def build(self):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.state = tf.placeholder(tf.float32, [None] + self.state_shape , name='state')
            self.next_state = tf.placeholder(tf.float32, [None] + self.state_shape , name='next_state')
            self.action = tf.placeholder(tf.uint8, [None] , name='action')
            self.done = tf.placeholder(tf.float32, [None] , name='done') 
            self.reward = tf.placeholder(tf.float32, [None] , name='reward')
            self.training = tf.placeholder(tf.bool, name='training_flag')
            self.learning_rate = tf.placeholder(tf.float32, None , name='learning_rate') 
            self.weights = tf.placeholder(tf.float32, [None], name="weight") # for prioritized experience replay

            net = self.model(self.state, self.training, self.n_actions, scope='net')
            target_net = self.model(self.next_state, self.training, self.n_actions, scope='target_net')

            net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + '/net')
            target_net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + '/target_net')

            self.best_action = tf.argmax(net, 1) # calculates best action given state
            Q = tf.reduce_sum(net * tf.one_hot(self.action, self.n_actions), 1) #performs action a_t at state s_t
            
            # not currently resettable => TODO
            #self.streaming_Q, self.streaming_Q_update = tf.contrib.metrics.streaming_mean(tf.reduce_mean(self.Q))
            #tf.summary.histogram("Q Value", self.streaming_Q)
            self.Q_summary = tf.summary.histogram("Q Value", tf.reduce_mean(Q))
            

            if self.ddqn:
                net_ = self.model(self.next_state, self.training, self.n_actions, scope='net', reuse=True)
                next_best_action = tf.argmax(net_, 1)
                next_Q = tf.reduce_sum(target_net * tf.one_hot(next_best_action, self.n_actions), 1)
            else:
                next_best_action = tf.argmax(target_net, 1)
                next_Q = tf.reduce_sum(target_net * tf.one_hot(next_best_action, self.n_actions), 1)

            bellman = self.reward + self.gamma * next_Q * (1.0 - self.done)
            self.error = Q - tf.stop_gradient(bellman)
            

            
            # self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.net_variables]) * self.weight_decay
            loss = self.loss_func(self.error) #+ self.l2_loss

            if isinstance(self.buffer, PrioritizedHistory):
                loss = tf.reduce_sum(self.weights * loss)
            else:
                loss = tf.reduce_sum(loss) #mean? LR should take care of it
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) #MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Updates batch norm layer, if exists

            if self.clip_grad is not None:
                gradients = optimizer.compute_gradients(loss, net_variables)
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, self.clip_grad), var)
                with tf.control_dependencies(extra_update_ops):
                    self.train = optimizer.apply_gradients(gradients)
            else:
                with tf.control_dependencies(extra_update_ops):
                    self.train = optimizer.minimize(loss, var_list = net_variables)

            # not currently resettable => TODO
            #self.streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(self.loss)
            #tf.summary.scalar('Loss', self.streaming_loss)
            self.loss_summary = tf.summary.scalar('Loss', tf.reduce_mean(loss))
            
            # tf.summary.scalar('Loss', self.loss)
            self.set_new_network = self.copy_to_target(net_variables, target_net_variables)
            self.summarize = tf.summary.merge([self.loss_summary, self.Q_summary])
            
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append(obs=state, actions=action, rewards=reward, new_obs=next_state, done=done)

    def store_reservoir(self, state, action):
        self.reservoir.append(obs=state, actions=action)

    def greedy_select(self, state, epsilon):

        if (epsilon > 0) and (np.random.uniform() <= epsilon):
            action = np.random.choice(np.arange(self.n_actions))
        else:
            action = self.sess.run(self.best_action, {self.state: state, self.training:False } )[0]

        return [action]

    def train_step(self, learning_rate_schedule, beta_schedule=None):
        # Copy the QNetwork weights to the Target QNetwork.
        if self.epoch == 0:
            self.sess.run(self.set_new_network)

        for batch_num in np.arange(self.batches_per_epoch):
            # Sample experience from replay memory
            if isinstance(self.buffer, PrioritizedHistory): 
                obs, act, rew, new_obs, done, weights, idxs  = self.buffer.sample(self.batch_size, is_sparse=self.is_sparse, beta=beta_schedule.value(self.epoch))
            else:
                obs, act, rew, new_obs, done  = self.buffer.sample(self.batch_size, self.gamma, is_sparse=self.is_sparse)

            # Perform training
            #_,_,_ = self.sess.run([self.streaming_loss_update, self.streaming_Q_update, self.train],
            if isinstance(self.buffer, PrioritizedHistory): 
                td_errors,_ = self.sess.run([self.error, self.train],
                                  { self.state: obs,
                                    self.next_state: new_obs,
                                    self.action: act,
                                    self.done: done,
                                    self.reward: rew,
                                    self.training: True,
                                    self.learning_rate: learning_rate_schedule.value(self.epoch),
                                    self.weights: weights} )

            
                new_priorities = np.abs(td_errors) + self.buffer.priority_eps
                self.buffer.update_priorities(idxs, new_priorities)

            else:
                _ = self.sess.run([self.train],
                                  { self.state: obs,
                                    self.next_state: new_obs,
                                    self.action: act,
                                    self.done: done,
                                    self.reward: rew,
                                    self.training: True,
                                    self.learning_rate: learning_rate_schedule.value(self.epoch)} )

        if isinstance(self.buffer, PrioritizedHistory): 
            to_write = self.sess.run(self.summarize, { self.state: obs, self.next_state: new_obs, self.action: act, self.done: done, self.reward: rew, self.training: True, self.learning_rate: learning_rate_schedule.value(self.epoch), self.weights: weights} )
        else:
            to_write = self.sess.run(self.summarize, { self.state: obs, self.next_state: new_obs, self.action: act, self.done: done, self.reward: rew, self.training: True, self.learning_rate: learning_rate_schedule.value(self.epoch)} )        
        
        self.summary_writer.add_summary(to_write, self.epoch)
        
        #self.summary_writer.flush() # Dont flush here; flush in training loop

        if (self.epoch > 0) and (self.epoch % self.update_freq == 0):
            self.sess.run(self.set_new_network)
        
        self.epoch += 1


class SelfPlay(DQN):
    def __init__(self,
                 sess,
                 model,
                 state_shape, 
                 writer,
                 n_actions=4, 
                 batch_size=32,
                 gamma=.99,
                 update_freq=1000,
                 ddqn=True, # double dqn
                 buffer_size = 10000,
                 clip_grad = None,
                 batches_per_epoch = 1,
                 is_sparse = True,
                 use_priority = False,
                 priority_alpha = .5,
                 priority_eps = 1e-8,
                 _id = '0',
                 policy_batch_size=32,
                 reservoir_buffer_size=100000
                 ):
    
        super(SelfPlay, self).__init__(sess,
                                         model,
                                         state_shape, 
                                         writer,
                                         n_actions, 
                                         batch_size,
                                         gamma,
                                         update_freq,
                                         ddqn,
                                         buffer_size,
                                         clip_grad,
                                         batches_per_epoch,
                                         is_sparse,
                                         use_priority,
                                         priority_alpha,
                                         priority_eps,
                                         _id)

        self.avg_policy_scope = 'avg_policy_%s' % self.id 
        self.avg_policy_batch_size = policy_batch_size
        self.policy_epoch = 0
        self.reservoir = Reservoir(1.1, reservoir_buffer_size)
        self.build_average_policy_network()

    def select_from_policy(self, state, epsilon, eta):
        is_greedy = False
        
        r = np.random.uniform()
        if (eta > 0) and (r <= eta): 
            is_greedy = True
            action = self.greedy_select(state, epsilon)
        else:
            action = self.average_policy_select(state)
            

        return action, is_greedy

    def average_policy_select(self, obs):
        policy = self.sess.run(tf.nn.softmax(self.logits), {self.policy_state: obs, self.training:False } )[0]
        action = np.random.choice(range(self.n_actions), p=policy)

        return [action]

    def build_average_policy_network(self):
        with tf.variable_scope(self.avg_policy_scope, reuse=self.reuse):
            self.policy_state = tf.placeholder(tf.float32, [None] + self.state_shape , name='state')
            self.policy_action = tf.placeholder(tf.int32, [None] , name='action')
            self.policy_learning_rate = tf.placeholder(tf.float32, None , name='learning_rate') 
            self.policy_training = tf.placeholder(tf.bool, name='training_flag')

            self.logits = self.model(self.policy_state, self.policy_training, self.n_actions, scope='net')            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.policy_action, logits=self.logits)
            loss = tf.reduce_mean(loss) 
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_learning_rate) #MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)
            
            self.avg_policy_train = optimizer.minimize(loss)

            self.avg_policy_loss_summary = tf.summary.scalar('Policy Loss', tf.reduce_mean(loss))
            
            self.avg_policy_summarize = tf.summary.merge([self.avg_policy_loss_summary])

    def avg_policy_train_step(self, learning_rate_schedule):


        for batch_num in np.arange(self.batches_per_epoch):
            # Sample experience from replay memory
            obs, act  = self.reservoir.sample(self.avg_policy_batch_size, is_sparse=self.is_sparse)

            # Perform training
            #_,_,_ = self.sess.run([self.streaming_loss_update, self.streaming_Q_update, self.train],
            if obs is not None:
                _ = self.sess.run([self.avg_policy_train],
                                  { self.policy_state: obs,
                                    self.policy_action: act,
                                    self.policy_training: True,
                                    self.policy_learning_rate: learning_rate_schedule.value(self.policy_epoch)} )
            else:
                return -1

        
        to_write = self.sess.run(self.avg_policy_summarize,  { self.policy_state: obs,
                                    self.policy_action: act,
                                    self.policy_training: True,
                                    self.policy_learning_rate: learning_rate_schedule.value(self.policy_epoch)} 
                                )
        
        self.summary_writer.add_summary(to_write, self.policy_epoch)
        
        self.policy_epoch += 1

