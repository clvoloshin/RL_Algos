import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten, fully_connected
from tensorflow.contrib.framework import arg_scope
import numpy as np
import pdb

def conv_layer(x, filter, kernel, stride, padding='SAME', bias = False, scope="conv", activation = None):
    with tf.variable_scope(scope):
        network = tf.layers.conv2d(inputs=x, 
                                   filters=filter, 
                                   kernel_size=kernel, 
                                   strides=stride, 
                                   padding=padding,
                                   use_bias = bias,
                                   kernel_initializer=tf.random_normal_initializer(stddev=.01),
                                   activation = activation)
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

def linear(x, units, scope, bias=False, activation = tf.nn.relu) :
    with tf.variable_scope(scope):
        return tf.layers.dense(x, units, activation = activation, use_bias = bias, kernel_initializer=tf.random_normal_initializer(stddev=.01))


def create_residual(n_resid_blocks):
    return lambda *args, **kwargs: Residual(n_resid_blocks,*args, **kwargs)

def create_basic(n_conv_layers, transpose=True):
    return lambda *args, **kwargs: Basic(n_conv_layers, transpose=transpose, *args, **kwargs)

def Basic(n_conv_layers, input_x, training, n_actions, scope, reuse=False, transpose=True):
    with tf.variable_scope(scope, reuse=reuse):
        
        with tf.variable_scope('input'):
            if transpose:
                x = tf.transpose(input_x, [0,2,3,1]) # NCHW to NHWC
            else:
                x = input_x
            
        with tf.variable_scope('conv_layer1'):
            conv_layer_1 = conv_layer(x, filter=16, kernel=[5, 5], stride=1, scope='conv_1', bias=True, activation=tf.nn.leaky_relu)

        with tf.variable_scope('conv_layer2'):
            conv_layer_2 = conv_layer(conv_layer_1, filter=16, kernel=[3, 3], stride=1, scope='conv_2', bias=True, activation=tf.nn.leaky_relu)

        with tf.variable_scope('out'):
            # feature_reduction = conv_layer(conv_layer_2, filter=1, kernel=[1, 1], stride=1, scope='out'+'_conv1', activation=tf.nn.relu)
            reshaped = tf.reshape(conv_layer_2, [-1, int(np.prod(conv_layer_2.shape[1:]))])

            pre = linear(reshaped, 64, bias=True, scope='out_linear1', activation=tf.nn.leaky_relu)
            out = linear(pre, n_actions, bias=True, scope='final', activation = None)

        return out


def Residual(n_resid_blocks, input_x, training, n_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        
        input_x = first_layer(input_x, training, scope='first_layer')

        for i in range(n_resid_blocks):
            input_x = residual_block(input_x, training, scope='residual_block_%s' % i)

        return value_head(input_x, training, 'value', n_actions)
        

def first_layer(x, training, scope):
    with tf.variable_scope(scope):
        x = tf.transpose(x, [0,2,3,1]) # NCHW to NHWC
        x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, scope=scope+'_conv1')
        x = batch_normalization(x, training=training, scope=scope+'_batch1')
        x = relu(x)

        return x

def residual_block(input_x, training, scope):
    with tf.variable_scope(scope):
        x = conv_layer(input_x, filter=64, kernel=[3, 3], stride=1, scope=scope+'_conv1')
        x = batch_normalization(x, training=training, scope=scope+'_batch1')
        x = relu(x)
        x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, scope=scope+'_conv2')
        x = batch_normalization(x, training=training, scope=scope+'_batch2')
        input_x = relu(x + input_x)

        return input_x

def value_head(x, training, scope, out_shape = 1): #Q value
    with tf.variable_scope(scope):
        x = conv_layer(x, filter=1, kernel=[1, 1], stride=1, scope=scope+'_conv1')
        x = batch_normalization(x, training=training, scope=scope+'_batch1')
        x = relu(x)

        x = tf.reshape(x, [-1, int(np.prod(x.shape[1:]))])

        x = linear(x, 256, scope=scope+'_linear1')
        x = relu(x)
        x = linear(x, out_shape, scope=scope+'_out')

        return x


    # def policy_head(self, x, scope, n_actions):
    #     with tf.variable_scope(scope):
    #         x = conv_layer(x, filter=2, kernel=[1, 1], stride=1, scope=scope+'_conv1')
    #         x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
    #         x = relu(x)

    #         x = tf.reshape(x, [-1, int(np.prod(x.shape[1:]))])
    #         x = linear(x, n_actions, scope=scope+'_out')

    #         return x

    

class Network(object):
    def __init__(self, x, n_actions, training, scope = 'main', reuse = False, just_value = True):
        self.training = training
        self.n_actions = n_actions
        self.reuse = reuse
        self.scope = scope
        if just_value:
            self.model = self.build_value(x)
        else:
            self.model = self.build_combined(x)

    

    def build_combined(self, input_x):
        input_x = self.first_layer(input_x, scope='first_layer')

        for i in range(2):
            input_x = self.residual_block(input_x, scope='residual_block_%s' % i)

        policy = self.policy_head(input_x, 'policy', self.n_actions)
        value = self.value_head(input_x, 'value')

        return policy, value

    def build_value(self, input_x):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            
            input_x = self.first_layer(input_x, scope='first_layer')

            for i in range(2):
                input_x = self.residual_block(input_x, scope='residual_block_%s' % i)

            value = self.value_head(input_x, 'value', self.n_actions)

            return value



class Policy(object):
    def __init__(self, env):
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

    @staticmethod
    def copy_to_target_network(source_network, target_network):
        target_network_update = []
        for v_source, v_target in zip(source_network.variables(), target_network.variables()):
            # this is equivalent to target = source
            update_op = v_target.assign(v_source)
            target_network_update.append(update_op)
        return tf.group(*target_network_update)

        # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

