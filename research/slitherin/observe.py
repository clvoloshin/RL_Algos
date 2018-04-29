import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #Should ideally check for GPU; assume exists for now
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pdb
import time
import numpy as np
import tensorflow as tf
import gym
import gym_snake
import scipy.sparse as sparse
from copy import deepcopy
from utils.history import History
from utils.action_policy_network import create_residual, create_basic
from utils.mcts.mcts import *
from utils.mcts.graph import Node
from utils.dqn import DQN
from utils.monitor import Monitor
from utils.schedules import LinearSchedule, PiecewiseSchedule

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

def run(**kwargs):
    '''
    Setup TF, gym environment, etc.
    '''

    
    logdir=kwargs['logdir']
    seed=kwargs['seed']
    headless=kwargs['headless']

    if headless:
        import matplotlib

    ################################################################
    # SEEDS
    ################################################################
    tf.set_random_seed(seed)
    np.random.seed(seed)

    
    ################################################################
    # SETUP GYM + RL ALGO
    ################################################################
    env = gym.make('snake-v0') # Make the gym environment
   

    ################################################################
    # TF BOILERPLATE
    ################################################################

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    with tf.Session() as sess:
        network = DQN( 
                     sess,
                     create_basic(3, transpose=True),
                     [1,env.world.screen_width,env.world.screen_height], 
                     None,
                     n_actions=4, 
                     batch_size=None,
                     gamma=.99,
                     update_freq=None,
                     ddqn=True, # double dqn
                     buffer_size = None,
                     clip_grad = None,
                     batches_per_epoch = None,
                     is_sparse = False
                     )

        monitor = Monitor(os.path.join(logdir,'test_gifs'))
        # summary_writer = tf.summary.FileWriter(logdir) 

        ## Load model from where you left off
        ## Does not play nice w/ plots in tensorboard at the moment
        ## TODO: FIX
        if True:
            try:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(logdir)
                pdb.set_trace()
                saver.restore(sess,ckpt.model_checkpoint_path)
                iteration_offset = int(ckpt.model_checkpoint_path.split('-')[-1].split('.')[0])
            except:
                print ('Failed to load. Starting from scratch')
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                iteration_offset = 0   
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            iteration_offset = 0

        pdb.set_trace()
        ################################################################
        # Fill Buffer
        ################################################################

        tic = time.time()
        total_timesteps = 0

        for iteration in range(1):
            _ = env.reset()
            obs = env.render('rgb_array', headless = headless).astype(float)
            obs /= obs.max()
            obs = rgb2gray(obs)

            done_n = np.array([False]*env.n_actors)
            steps = 0
            while not done_n.all():

                if True:
                    if (not viewer) and (not headless):
                        from gym.envs.classic_control import rendering
                        viewer = rendering.SimpleImageViewer()

                    rgb = env.render('rgb_array', headless = headless)
                    scaler = 10
                    rgb=repeat_upsample(rgb,scaler,scaler)

                    if not headless:
                        
                        viewer.imshow(rgb)
                        time.sleep(.01)

                    monitor.add(rgb, iteration, iteration)

                last_obs = obs
                acts = network.greedy_select([[last_obs]], 1.) 
                acts = [str(x) for x in acts]
      
                # Next step
                _, reward_n, done_n = env.step(acts[-1])
                obs = env.render('rgb_array', headless = headless).astype(float)
                obs /= obs.max()
                obs = rgb2gray(obs)

                steps += 1

            monitor.make_gifs(iteration)
            
            # Log diagnostics
            # returns = history.get_total_reward_per_sequence()
            # ep_lengths = history.get_timesteps()

            

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--headless', '-hless', action='store_true')
    args = parser.parse_args()

    if not(os.path.exists('output')):
        os.makedirs('output')
    logdir = os.path.join('output', 'SnakeEnv', str(args.seed))
    if os.path.exists(logdir):
        # message =  'You are about to remove a previous run\'s directory.'
        # message2 = ' You must specify the appropriate flag to do so.'
        # assert args.remove_prev_runs, message + message2
        # shutil.rmtree(logdir)
        # os.makedirs(logdir)
        pass
    else:
        os.makedirs(logdir)

    print('Seed: %d'%args.seed)
    run(logdir=logdir,
        seed=args.seed,
        headless=args.headless)

if __name__ == "__main__":
    main()