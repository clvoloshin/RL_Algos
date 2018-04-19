import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import pdb

def conv_layer(input, filter, kernel, stride, padding='SAME', scope="conv"):
    with tf.name_scope(scope):
        network = tf.layers.conv2d(inputs=input, 
                                   use_bias=False, 
                                   filters=filter, 
                                   kernel_size=kernel, 
                                   strides=stride, 
                                   padding=padding)
        return network

def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def relu(x):
    return tf.nn.relu(x)

def linear(x, n_actions, scope) :
    with tf.name_scope(scope):
        return tf.layers.dense(inputs=x, use_bias=False, units=n_actions, name=scope)

class Network(object):
    def __init__(self, x, n_actions, training):
        self.training = training
        self.n_actions = n_actions
        self.model = self.build_network(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = tf.transpose(x, [0,2,3,1]) # NCHW to NHWC
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, scope=scope+'_conv1')
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)

            return x

    def residual_block(self, input_x, scope):
        with tf.name_scope(scope):
            x = conv_layer(input_x, filter=64, kernel=[3, 3], stride=1, scope=scope+'_conv1')
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, scope=scope+'_conv2')
            x = batch_normalization(x, training=self.training, scope=scope+'_batch2')
            x = relu(x + input_x)

            return x

    def policy_head(self, x, scope, n_actions):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=2, kernel=[1, 1], stride=1, scope=scope+'_conv1')
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)

            x = tf.reshape(x, [-1, int(np.prod(x.shape[1:]))])
            x = linear(x, n_actions, scope=scope+'_out')

            return x

    def value_head(self, x, scope): #Q value
        with tf.name_scope(scope):
            x = conv_layer(x, filter=1, kernel=[1, 1], stride=1, scope=scope+'_conv1')
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)

            x = tf.reshape(x, [-1, int(np.prod(x.shape[1:]))])

            x = linear(x, 256, scope=scope+'_linear1')
            x = relu(x)
            x = linear(x, 1, scope=scope+'_out')

            return x

    def build_network(self, input_x):
        input_x = self.first_layer(input_x, scope='first_layer')

        for i in range(2):
            input_x = self.residual_block(input_x, scope=str(i))

        policy = self.policy_head(input_x, 'policy', self.n_actions)
        value = self.value_head(input_x, 'value')

        return policy, value



class Policy(object):
    def __init__(self, env):
        '''
        DQN
        '''

        self.weight_decay = 1e-4
        self.momentum = .9
        self.obs = tf.placeholder(tf.float32, shape=[None, env.n_actors*2+2, env.screen_width, env.screen_width ])
        self.target_policy = tf.placeholder(tf.float32, shape=[None, len(env.world.action_space)])
        self.target_Q = tf.placeholder(tf.float32, shape=[None])

        self.training_flag = tf.placeholder(tf.bool)


        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.policy_logits, self.pred_Q = Network(self.obs, len(env.world.action_space),training=self.training_flag).model
        self.softmax_policy = tf.nn.softmax(self.policy_logits)

        policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_policy, logits=self.policy_logits))
        value_loss = tf.nn.l2_loss(self.target_Q - self.pred_Q)
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * self.weight_decay
        
        self.loss = value_loss + policy_loss + l2_loss

        self.sample_action = tf.reshape(tf.multinomial(self.policy_logits, 1), [-1])


        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)
        self.train = self.optimizer.minimize(self.loss)

        # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


