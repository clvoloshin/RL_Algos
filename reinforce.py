import numpy as np
import tensorflow as tf
import gym
import os
import time
import inspect
import scipy.signal
from multiprocessing import Process
import json
import pdb
import shutil

def build_nn(input_placeholder,
             output_size,
             scope,
             num_layers = 2,
             num_units = 64,
             activation = tf.tanh,
             output_activation = None
            ):
    '''
    input_placeholder: TF placeholder representing data fed into the NN
    outputs_size: A number. This will represent the number of possible actions that can be taken (in discrete case)
    scope: Scope name for TF
    num_layers: Number of hidden layers.
                    Default: 2 layers
    num_units: Number of nodes per hidden layer. Not including the output layer.
                    Default: 64 units
    activation: Activation after each hidden layer. Not including the output layer activation.
                    Default: tanh activation
    output_activation: The output layer activation.
                    Default: None (ie. linear)
    '''
    
    with tf.variable_scope(scope):
        output = input_placeholder
        for layer in range(num_layers):
            output = tf.layers.dense(inputs=output, 
                                     units=num_units, 
                                     activation = activation)
        output = tf.layers.dense(inputs=output, 
                                 units=output_size, 
                                 activation = output_activation)
        return output

class REINFORCE():
    def __init__(self, env, args, scope = 'policy'):
        '''
        REINFORCE algorithm
        env: Gym environment from OpenAi
        scope: TF scope
        args: dictionary. See command line args for instructions.
                Req args: 
                    (1) num_units (int)
                    (2) num_layers (int)
                    (3) learning rate (float)
                    (4) use_baseline (boolean)
        '''
        with tf.variable_scope(scope):
            discrete = isinstance(env.action_space, gym.spaces.Discrete)
            self.obs_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]
            ################################################################
            #
            # obs: Observations                   Shape = (batch_size (n), dimension of observation space)
            # target_actions: "ideal" actions     Shape = (batch_size (n)) if discrete else
            #                                             (batch_size (n), dimension of action space)
            # advantages: Q values or Q-baseline  Shape = (batch_size (n))
            #    = sum_{t'=0}^T gamma^t' r_{t'}
            ################################################################
            self.obs = tf.placeholder(shape=[None, self.obs_dim], name="obs", dtype=tf.float32)

            if discrete:
                self.target_actions = tf.placeholder(shape=[None], name="action", dtype=tf.int32)
            else:
                # TODO: deal w/ continuous space
                self.target_actions = tf.placeholder(shape=[None, self.action_dim], name="action", dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], name="Q", dtype=tf.float32)


            if args['use_baseline']:
                self.setup_baseline(args)


            # Represents the output represents a policy prediction (logits) for each input observation
            self.pred_actions = build_nn(self.obs,
                                         self.action_dim,
                                         'logits',
                                         num_layers = args['num_layers'],
                                         num_units = args['num_units']
                                        )

            # sample an action from the policy
            self.sample_action = tf.reshape(tf.multinomial(self.pred_actions, 1), [-1])

            
            self.negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.target_actions, logits=self.pred_actions)
            
            ###
            # Just negative_likelihoods is normal maximum likelihood estimation.
            # Need to multuply each likelihood by the Q value to get the policy gradient
            # \nabla J = E[\nabla \log \pi * r] => \nabla J = 1/N \sum\sum likelihoods * Q
            ###
            self.weighted_negative_likelihoods = tf.multiply(self.negative_likelihoods, self.advantages)

            self.loss = tf.reduce_mean(self.weighted_negative_likelihoods)

            self.update = tf.train.AdamOptimizer(args['learning_rate']).minimize(self.loss)

    def setup_baseline(self, args):
        '''
        Creates NN baseline for policy method.
        args: dictionary. See command line args for instructions.
                Req args: 
                    (1) num_units
                    (2) num_layers
                    (3) learning rate
        '''

        self.baseline = build_nn(self.obs,
                                 1, # baseline is a scalar
                                 'baseline',
                                 num_layers = args['num_layers'], #TODO: make variables in yaml file instead of command line
                                 num_units = args['num_units']    
                                )

        self.target_q_values = tf.placeholder(shape=[None], name="target_q", dtype=tf.float32)
        self.baseline_loss = tf.nn.l2_loss(self.baseline - self.target_q_values)

        self.update_baseline = tf.train.AdamOptimizer(args['learning_rate']).minimize(self.baseline_loss)


class History():
    '''
        Keeps track of the history of observations, actions, and rewards
    '''
    def __init__(self):
        self.sequences = []
        self.timesteps = []
        self.total_timesteps = 0

    def append(self,**kwargs):
        '''
        Adds newest sequence to the history
        
        Requires:
            (1) obs: np.array of observations
            (2) actions: np.array of actions
            (3) rewards: np.array of rewards
        '''
        self.sequences.append(
                            {'observations': kwargs['obs'],
                            'actions': kwargs['actions'],
                            'rewards': kwargs['rewards']}
                             )
        self.timesteps.append(kwargs['obs'].shape[0])
        self.total_timesteps += self.timesteps[-1]

    def get_all_obs(self):
        return np.vstack([path['observations'] for path in self.sequences])

    def get_all_actions(self):
        return np.hstack([path['actions'] for path in self.sequences])

    def get_all_rewards(self, discount, use_reward_to_go = False):
        if use_reward_to_go:
            # If reward_to_go then use the future rewards at every step in the path:
            # Q(t) = \sum_{i=t}^T x[i] * gamma^{i-t}, not a function of t
            return np.hstack([self.discount_rewards(path['rewards'], discount) for path in self.sequences])
        else:
            # If not reward_to_go then use the total reward at every step in the path:
            # Q = \sum_{i=0}^T x[i] * gamma^i, not a function of t
            return np.hstack([[self.discount_rewards(path['rewards'], discount)[0]] * len(path['rewards']) for path in self.sequences])

    def get_total_reward_per_sequence(self):
        return [path['rewards'].sum() for path in self.sequences]

    def get_total_timesteps(self):
        return self.total_timesteps

    def get_timesteps(self):
        return self.timesteps

    def reset(self):
        self.__init__()

    @staticmethod
    def discount_rewards(x, gamma): 
        '''
        x: vector of shape (T,1)
        gamma: scalar. Represents the discount
        Returns [\sum_{i=0}^T x[i] * gamma^i, \sum_{i=1}^T x[i] gamma^{i-1}, ...]
        '''
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def run(**kwargs):
    '''
    Setup TF, gym environment, etc.
    '''

    env_name=kwargs['env_name']
    iterations=kwargs['iterations']
    discount=kwargs['discount']
    batch_size=kwargs['batch_size']
    max_seq_length=kwargs['max_seq_length']
    learning_rate=kwargs['learning_rate']
    animate=kwargs['animate']
    logdir=kwargs['logdir']
    seed=kwargs['seed']
    load_model=kwargs['load_model']
    use_baseline=kwargs['use_baseline']
    use_reward_to_go=kwargs['use_reward_to_go']


    ################################################################
    # SEEDS
    ################################################################
    tf.set_random_seed(seed)
    np.random.seed(seed)

    
    ################################################################
    # SETUP GYM + RL ALGO
    ################################################################
    env = gym.make(env_name) # Make the gym environment
    maximum_number_of_steps = max_seq_length or env.spec.max_episode_steps # Maximum length for episodes

    pi = REINFORCE(env, kwargs)

    ################################################################
    # TF BOILERPLATE
    ################################################################

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    saver = tf.train.Saver(max_to_keep=1)
    summary_writer = tf.summary.FileWriter(logdir) 

    with tf.Session() as sess:

        ## Load model from where you left off
        ## Does not play nice w/ plots in tensorboard at the moment
        if load_model == True:
            try:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(logdir)
                saver.restore(sess,ckpt.model_checkpoint_path)
                iteration_offset = int(ckpt.model_checkpoint_path.split('-')[-1].split('.')[0])
            except:
                print ('Failed to load. Starting from scratch')
                sess.run(tf.global_variables_initializer())
                iteration_offset = 0   
        else:
            sess.run(tf.global_variables_initializer())
            iteration_offset = 0

        summary_writer.add_graph(sess.graph)

        

        ################################################################
        # Train Loop
        ################################################################

        tic = time.time()
        total_timesteps = 0
        history = History()

        for iteration in range(iteration_offset, iteration_offset + iterations):
            print('{0} Iteration {1} {0}'.format('*'*10, iteration))

            timesteps_in_iteration = 0
            history.reset()

            if (iteration % 10 == 0):
                saver.save(sess,os.path.join(logdir,'model-'+str(iteration)+'.cptk'))
                print ("Saved Model. Timestep count: %s" % iteration) 

            while True:
                obs = env.reset()
                observations, actions, rewards = [], [], []

                animate_episode = (history.get_total_timesteps()==0) and (iteration % 10 == 0) and animate

                done = False
                steps = 0

                # Runs policy, collects observations and rewards
                while not done:
                    if animate_episode:
                        env.render()
                        time.sleep(.05)

                    observations.append(obs)
                    action = sess.run(pi.sample_action, feed_dict={pi.obs: [obs]})
                    actions.append(action[0])

                    # Next step

                    obs, reward, done, info = env.step(actions[-1])
                    rewards.append(reward)
                    steps += 1

                    # terminate the collection of data if the controller shows stability
                    # for a long time. This is a good thing.
                    if steps > maximum_number_of_steps:
                        done = True

                history.append(obs=np.array(observations), 
                               actions=np.array(actions), 
                               rewards=np.array(rewards))
                
                if history.get_total_timesteps() > batch_size:
                    break

            total_timesteps += history.get_total_timesteps()


            if use_baseline:
                # Update Baseline
                # Update Policy
                baseline = sess.run(pi.baseline, feed_dict={pi.obs: history.get_all_obs()})

                # Advantage is Q-baseline
                advantage = history.get_all_rewards(discount, use_reward_to_go) - baseline.reshape([-1])

                sess.run([pi.update, pi.update_baseline], feed_dict={pi.obs: history.get_all_obs(),
                                                                     pi.target_actions: history.get_all_actions(),
                                                                     pi.advantages: advantage,
                                                                     pi.target_q_values: history.get_all_rewards(discount, use_reward_to_go) })
            else:
                # Update Policy
                # Without baseline, advantages are just Q values
                sess.run(pi.update, feed_dict={pi.obs: history.get_all_obs(),
                                               pi.target_actions: history.get_all_actions(),
                                               pi.advantages: history.get_all_rewards(discount, use_reward_to_go)})

            # Log diagnostics
            returns = history.get_total_reward_per_sequence()
            ep_lengths = history.get_timesteps()


            summary = tf.Summary()
            summary.value.add(tag='Average Return', simple_value=np.mean(returns))
            summary.value.add(tag='Std Return', simple_value=np.std(returns))
            summary.value.add(tag='Max Return', simple_value=np.max(returns))
            summary.value.add(tag='Min Return', simple_value=np.min(returns))
            summary.value.add(tag='Episode Length Mean', simple_value=np.mean(ep_lengths))
            summary.value.add(tag='Episode Length Std', simple_value=np.std(ep_lengths))
            summary.value.add(tag='Timesteps in Batch', simple_value=history.get_total_timesteps())
            summary.value.add(tag='Total Timesteps', simple_value=total_timesteps)
            summary_writer.add_summary(summary, iteration)
            summary_writer.flush()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--use_baseline', '-bl', action='store_true')
    parser.add_argument('--use_reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--iterations', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--sequence_length', '-seq', type=int, default=0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-num_exp', type=int, default=1)
    parser.add_argument('--num_layers', '-l', type=int, default=2)
    parser.add_argument('--num_units', '-u', type=int, default=32)
    parser.add_argument('--load_model', '-load', action='store_true')
    # parser.add_argument('--remove_prev_runs', '-rm', action='store_true')
    args = parser.parse_args()

    if not(os.path.exists('output')):
        os.makedirs('output')
    logdir = os.path.join('output', args.env_name, str(args.seed))
    if os.path.exists(logdir):
        # message =  'You are about to remove a previous run\'s directory.'
        # message2 = ' You must specify the appropriate flag to do so.'
        # assert args.remove_prev_runs, message + message2
        # shutil.rmtree(logdir)
        # os.makedirs(logdir)
        pass
    else:
        os.makedirs(logdir)

    max_seq_length = args.sequence_length if args.sequence_length > 0 else None

    print('Seed: %d'%args.seed)
    run(env_name=args.env_name,
        iterations=args.iterations,
        discount=args.discount,
        batch_size=args.batch_size,
        max_seq_length=max_seq_length,
        learning_rate=args.learning_rate,
        animate=args.animate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        logdir=logdir,
        seed=args.seed,
        num_layers=args.num_layers,
        num_units=args.num_units,
        load_model=args.load_model)

if __name__ == "__main__":
    main()