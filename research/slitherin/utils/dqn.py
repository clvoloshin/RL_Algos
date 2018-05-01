import tensorflow as tf
import numpy as np
import pdb
from history import History, PrioritizedHistory

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
                 priority_eps = 1e-8
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

        self.scope = 'main'
        self.reuse = None
        
        self.build()

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

            self.net = self.model(self.state, self.training, self.n_actions, scope='net')
            self.target_net = self.model(self.next_state, self.training, self.n_actions, scope='target_net')

            self.net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + '/net')
            self.target_net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + '/target_net')

            self.best_action = tf.argmax(self.net, 1) # calculates best action given state
            self.Q = tf.reduce_sum(self.net * tf.one_hot(self.action, self.n_actions), 1) #performs action a_t at state s_t
            
            # not currently resettable => TODO
            #self.streaming_Q, self.streaming_Q_update = tf.contrib.metrics.streaming_mean(tf.reduce_mean(self.Q))
            #tf.summary.histogram("Q Value", self.streaming_Q)
            tf.summary.histogram("Q Value", tf.reduce_mean(self.Q))
            

            if self.ddqn:
                self.net_ = self.model(self.next_state, self.training, self.n_actions, scope='net', reuse=True)
                self.next_best_action = tf.argmax(self.net_, 1)
                self.next_Q = tf.reduce_sum(self.target_net * tf.one_hot(self.next_best_action, self.n_actions), 1)
            else:
                self.next_best_action = tf.argmax(self.target_net, 1)
                self.next_Q = tf.reduce_sum(self.target_net * tf.one_hot(self.next_best_action, self.n_actions), 1)

            self.bellman = self.reward + self.gamma * self.next_Q * (1.0 - self.done)
            self.error = self.Q - tf.stop_gradient(self.bellman)
            

            
            self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.net_variables]) * self.weight_decay
            loss = tf.reduce_sum(self.loss_func(self.error)) #+ self.l2_loss

            if isinstance(self.buffer, PrioritizedHistory):
                self.loss = tf.reduce_mean(self.weights * loss)
            else:
                self.loss = loss
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) #MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)

            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Updates batch norm layer, if exists

            if self.clip_grad is not None:
                self.gradients = self.optimizer.compute_gradients(self.loss, self.net_variables)
                for i, (grad, var) in enumerate(self.gradients):
                    if grad is not None:
                        self.gradients[i] = (tf.clip_by_norm(grad, self.clip_grad), var)
                with tf.control_dependencies(self.extra_update_ops):
                    self.train = self.optimizer.apply_gradients(self.gradients)
            else:
                with tf.control_dependencies(self.extra_update_ops):
                    self.train = self.optimizer.minimize(self.loss, var_list = self.net_variables)

            # not currently resettable => TODO
            #self.streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(self.loss)
            #tf.summary.scalar('Loss', self.streaming_loss)
            tf.summary.scalar('Loss', tf.reduce_mean(self.loss))
            
            # tf.summary.scalar('Loss', self.loss)
            self.set_new_network = self.copy_to_target(self.net_variables, self.target_net_variables)
            self.summarize = tf.summary.merge_all()
            
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append(obs=state, actions=action, rewards=reward, new_obs=next_state, done=done)


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
                td_errors, loss,_ = self.sess.run([self.error, self.loss, self.train],
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
                loss,_ = self.sess.run([self.loss, self.train],
                                  { self.state: obs,
                                    self.next_state: new_obs,
                                    self.action: act,
                                    self.done: done,
                                    self.reward: rew,
                                    self.training: True,
                                    self.learning_rate: learning_rate_schedule.value(self.epoch)} )

        if isinstance(self.buffer, PrioritizedHistory): 
            to_write = self.sess.run(self.summarize, { self.weights: weights, self.state: obs, self.next_state: new_obs, self.action: act, self.done: done, self.reward: rew, self.training: True, self.learning_rate: learning_rate_schedule.value(self.epoch)} )
        else:
            to_write = self.sess.run(self.summarize, { self.state: obs, self.next_state: new_obs, self.action: act, self.done: done, self.reward: rew, self.training: True, self.learning_rate: learning_rate_schedule.value(self.epoch)} )        
        
        self.summary_writer.add_summary(to_write, self.epoch)
        
        #self.summary_writer.flush() # Dont flush here; flush in training loop

        if (self.epoch > 0) and (self.epoch % self.update_freq == 0):
            self.sess.run(self.set_new_network)
        
        pdb.set_trace()
        self.epoch += 1




