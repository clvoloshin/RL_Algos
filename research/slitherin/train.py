import pdb
import time
import numpy as np
import os
import tensorflow as tf
import gym
import gym_snake
import scipy.sparse as sparse
from copy import deepcopy
from utils.history import History
from utils.action_policy_network import create_residual
from utils.mcts.mcts import *
from utils.mcts.graph import Node
from utils.dqn import DQN
from utils.monitor import Monitor
from utils.schedules import LinearSchedule, PiecewiseSchedule

def get_data(obs):
    assert obs.shape[0] < 3
    data = obs#[:,:-1]
    if obs.shape[0] == 2:
        data = np.vstack([data, np.diff(data,axis=0)])
    else:
        data = np.vstack([data, np.array([[sparse.csr_matrix(data[0][0].shape) for _ in range(data.shape[1]) ]]) ])

    # only care about X_i and delta X_i
    data = data[-2:]

    player_idx = []
    for idx in range(data.shape[1]-1):
        idxs = np.array([False]*(data.shape[1]))
        idxs[idx] = True
        player_idx.append(np.hstack([data[:,idxs].reshape(-1), data[:,~idxs].T.reshape(-1)])) # Transpose so that you get column ordering rather than row ordering (C vs Fortran)

    return np.vstack(player_idx)

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

    iterations=kwargs['iterations']
    discount=kwargs['discount']
    batch_size=kwargs['batch_size']
    num_batches=kwargs['num_batches']
    max_seq_length=kwargs['max_seq_length']
    learning_rate=kwargs['learning_rate']
    animate=kwargs['animate']
    logdir=kwargs['logdir']
    seed=kwargs['seed']
    games_played_per_epoch=kwargs['games_played_per_epoch']
    load_model = False
    mcts_iterations=kwargs['mcts_iterations']
    batches_per_epoch=kwargs['batches_per_epoch']
    headless=kwargs['headless']
    update_freq=kwargs['update_freq']
    buffer_size=kwargs['buffer_size']

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
    maximum_number_of_steps = max_seq_length #or env.max_episode_steps # Maximum length for episodes
   

    ################################################################
    # TF BOILERPLATE
    ################################################################

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)

    summary_writers = []
    for idx in np.arange(env.n_actors):
        summary_writers.append(tf.summary.FileWriter(os.path.join(logdir,'tensorboard','snake_%s' % idx) ))

    summary_writers.append(tf.summary.FileWriter(os.path.join(logdir,'tensorboard','training_stats') ))    

    with tf.Session() as sess:
        network = DQN( 
                     sess,
                     create_residual(2),
                     [(env.world.number_of_snakes+1)*2, env.world.screen_width,env.world.screen_height], 
                     summary_writers[-1],
                     n_actions=4, 
                     batch_size=batch_size,
                     gamma=.99,
                     update_freq=update_freq,
                     ddqn=True, # double dqn
                     buffer_size = buffer_size,
                     clip_grad = None,
                     batches_per_epoch = batches_per_epoch
                     )

        monitor = Monitor(os.path.join(logdir,'gifs'))
        epsilon_schedule = LinearSchedule(20, 1.0, 0.0)
        learning_rate_schedule = PiecewiseSchedule([(0,1e-2),(1000,1e-3),(10000,1e-4)], outside_value=1e-4)

        saver = tf.train.Saver(max_to_keep=2)
        # summary_writer = tf.summary.FileWriter(logdir) 

        ## Load model from where you left off
        ## Does not play nice w/ plots in tensorboard at the moment
        ## TODO: FIX
        if load_model == True:
            try:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(logdir)
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

        summary_writers[0].add_graph(sess.graph)

        ################################################################
        # Train Loop
        ################################################################

        tic = time.time()
        total_timesteps = 0

        while not network.buffer.full():
            network.buffer.games_played += 1
            print 'Game number: %s. Buffer_size: %s' % (network.buffer.games_played, network.buffer.buffer_size)
            obs = env.reset()

            raw_observations = []
            raw_observations.append(np.array(obs))

            done_n = np.array([False]*env.n_actors)
            steps = 0
            length_alive = np.array([0] * env.n_actors)
            while not done_n.all():
                length_alive[env.world.idxs_of_alive_snakes] += 1
                ob = get_data(np.array(raw_observations)[-2:])
                acts = network.greedy_select(ob, epsilon_schedule.value(0)) 
                acts = [str(x) for x in acts]
      
                # Next step
                obs, reward_n, done_n = env.step(acts[-1])
                raw_observations.append(np.array(obs))
                steps += 1

                network.store(np.array(get_data(np.array(raw_observations)[-3:-1])), # state
                              np.array(acts), # action
                              np.array(reward_n), #rewards
                              np.array(get_data(np.array(raw_observations)[-2:])), #new state
                              np.array(done_n) #done
                              )

                # terminate the collection of data if the controller shows stability
                # for a long time. This is a good thing.
                if steps > maximum_number_of_steps:
                    done_n[:] = True

        print 'Filled Buffer'


        for iteration in range(iteration_offset, iteration_offset + iterations):
            print('{0} Iteration {1} {0}'.format('*'*10, iteration))
            network.buffer.soft_reset()
            timesteps_in_iteration = 0

            if (iteration % update_freq == 0):
                saver.save(sess,os.path.join(logdir,'model-'+str(iteration)+'.cptk'))
                print "Saved Model. Timestep count: %s" % iteration

            total_number_of_steps_in_iteration = 0

            while True:
                network.buffer.games_played += 1
                if (((network.buffer.games_played) % 10) == 0):
                    print 'Epoch: %s. Game number: %s' % (iteration, network.buffer.games_played)
                obs = env.reset()
                epsilon_schedule.reset()
                raw_observations = []
                raw_observations.append(np.array(obs))


                animate_episode = ((network.buffer.games_played-1)==0) and (iteration % update_freq == 0) and animate

                done_n = np.array([False]*env.n_actors)
                steps = 0

                # Runs policy, collects observations and rewards
                viewer = None

                length_alive = np.array([0] * env.n_actors)
                while not done_n.all():
                    if animate_episode:
                        if (not viewer) and (not headless):
                            from gym.envs.classic_control import rendering
                            viewer = rendering.SimpleImageViewer()

                        rgb = env.render('rgb_array', headless = headless)
                        scaler = 10
                        rgb=repeat_upsample(rgb,scaler,scaler)

                        if not headless:
                            
                            viewer.imshow(rgb)
                            time.sleep(.1)

                        monitor.add(rgb, iteration, network.buffer.games_played)


                    length_alive[env.world.idxs_of_alive_snakes] += 1

                    
                    ob = get_data(np.array(raw_observations)[-2:])

                    # Control the exploration
                    if network.buffer.games_played < (games_played_per_epoch/2):
                        # play half of the games w/ full greedy strategy
                        acts = network.greedy_select(ob, epsilon_schedule.value(1.)) # full greedy 
                    else:
                        # play other half w/ epsilon greedy
                        acts = network.greedy_select(ob, epsilon_schedule.next()) # epsilon greedy

                    acts = [str(x) for x in acts]
          
                    # Next step
                    obs, reward_n, done_n = env.step(acts[-1])
                    raw_observations.append(np.array(obs))
                    steps += 1

                    network.store(np.array(get_data(np.array(raw_observations)[-3:-1])), # state
                                  np.array(acts), # action
                                  np.array(reward_n), #rewards
                                  np.array(get_data(np.array(raw_observations)[-2:])), #new state
                                  np.array(done_n) #done
                                  )

                    # terminate the collection of data if the controller shows stability
                    # for a long time. This is a good thing.
                    if steps > maximum_number_of_steps:
                        done_n[:] = True

                total_number_of_steps_in_iteration += steps

                if viewer:
                    viewer.close()

                if network.buffer.games_played >= games_played_per_epoch:
                    break

            monitor.make_gifs(iteration)
            network.train_step(learning_rate_schedule)
            

            for count, writer in enumerate(summary_writers):
                writer.flush()
            #     summary = tf.Summary()
            #     summary.value.add(tag='Steps Alive', simple_value=(network.buffer.get_last_lengths_of_games()[count::env.n_actors]).mean())
            #     writer.add_summary(summary, iteration)
            #     

            # Log diagnostics
            # returns = history.get_total_reward_per_sequence()
            # ep_lengths = history.get_timesteps()

            

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--iterations', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--num_batches', '-n_batch', type=int, default=10) # num batch/epoch
    parser.add_argument('--sequence_length', '-seq', type=int, default=200)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-num_exp', type=int, default=1)
    parser.add_argument('--games_played_per_epoch', '-gpe', type=int, default=10)
    parser.add_argument('--mcts_iterations', '-mcts', type=int, default=1500)
    parser.add_argument('--batches_per_epoch', '-bpe', type=int, default=1)
    parser.add_argument('--buffer_size', '-bs', type=int, default=10000)
    parser.add_argument('--update_freq', '-uf', type=int, default=10000)
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

    max_seq_length = args.sequence_length if args.sequence_length > 0 else None

    print('Seed: %d'%args.seed)
    run(iterations=args.iterations,
        discount=args.discount,
        batch_size=args.batch_size,
        max_seq_length=max_seq_length,
        learning_rate=args.learning_rate,
        animate=args.animate,
        logdir=logdir,
        seed=args.seed,
        num_batches=args.num_batches,
        games_played_per_epoch=args.games_played_per_epoch,
        mcts_iterations=args.mcts_iterations,
        batches_per_epoch=args.batches_per_epoch,
        headless=args.headless,
        update_freq=args.update_freq,
        buffer_size=args.buffer_size)

if __name__ == "__main__":
    main()