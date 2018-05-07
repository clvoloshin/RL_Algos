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
from utils.dqn import DQN, SelfPlay
from utils.monitor import Monitor
from utils.schedules import LinearSchedule, PiecewiseSchedule

# def get_data(obs):
#     assert obs.shape[0] < 3
#     data = obs#[:,:-1]
#     if obs.shape[0] == 2:
#         data = np.vstack([data, np.diff(data,axis=0)])
#     else:
#         data = np.vstack([data, np.array([[sparse.csr_matrix(data[0][0].shape) for _ in range(data.shape[1]) ]]) ])

#     # only care about X_i and delta X_i
#     data = data[-2:]

#     player_idx = []
#     for idx in range(data.shape[1]-1):
#         idxs = np.array([False]*(data.shape[1]))
#         idxs[idx] = True
#         player_idx.append(np.hstack([data[:,idxs].reshape(-1), data[:,~idxs].T.reshape(-1)])) # Transpose so that you get column ordering rather than row ordering (C vs Fortran)

#     return np.vstack(player_idx)

def get_data(obs, actor):
    assert ((len(obs) % 2) == 1)
    return obs[actor*2:((actor+1)*2)] + obs[:actor*2] + obs[((actor+1)*2):]



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
    use_priority=kwargs['use_priority']
    policy_batch_size=kwargs['policy_batch_size']

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
    env = gym.make('snake-v1') # Make the gym environment
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
        
        networks = []

        for i in range(env.n_actors):
            networks.append( SelfPlay( 
                     sess,
                     create_basic([128,128,256], transpose=True),
                     [(env.n_actors)*2 + 1, env.world.screen_width,env.world.screen_height], 
                     summary_writers[-1],
                     n_actions=4, 
                     batch_size=batch_size,
                     gamma=.99,
                     update_freq=update_freq,
                     ddqn=True, # double dqn
                     buffer_size = buffer_size,
                     clip_grad = None,
                     batches_per_epoch = batches_per_epoch,
                     is_sparse = True,
                     use_priority=use_priority,
                     _id = i,
                     policy_batch_size = policy_batch_size
                     ) ) 

        monitor = Monitor(os.path.join(logdir,'gifs'))
        epsilon_schedule = LinearSchedule(iterations*5/10, .5, 0.01)
        eta_schedule = LinearSchedule(iterations*7/10, 0.2, 0.1)
        if use_priority:
            beta_schedule = LinearSchedule(iterations, 0.4, 1.)
        learning_rate_schedule = PiecewiseSchedule([(0,1e-3),(20000,5e-4),(50000,1e-4)], outside_value=1e-4)
        policy_learning_rate_schedule = PiecewiseSchedule([(0,1e-3),(4000,5e-4),(15000,1e-4)], outside_value=1e-4)

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

        while not all([network.buffer.full(N=int(buffer_size/2.)) for network in networks]):
            networks[0].buffer.games_played += 1
            print 'Game number: %s. Buffer_sizes: %s' % (networks[0].buffer.games_played, [network.buffer.buffer_size for network in networks])
            obs = env.reset()

            done_n = np.array([False]*env.n_actors)
            steps = 0
            length_alive = np.array([0] * env.n_actors)
            viewer = None
            while not done_n.all():

                length_alive[env.world.idxs_of_alive_snakes] += 1
                last_obs = obs

                acts = []
                for i, network in enumerate(networks):
                    act = network.greedy_select(np.array([[x.A for x in get_data(last_obs, i)]]), 1.) 
                    acts += [str(act[0])]
      
                # Next step
                obs, reward_n, done_n = env.step(acts)
                steps += 1

                for i in env.world.idxs_of_alive_snakes:
                    networks[i].store(np.array(get_data(last_obs, i)), # state
                                      np.array(acts[i]), # action
                                      np.array(reward_n[i]), #rewards
                                      np.array(get_data(obs, i)), #new state
                                      np.array(done_n[i]) #done
                                      )

                    networks[i].store_reservoir(np.array(get_data(last_obs, i)), # state
                                                        np.array(int(acts[i])))

                # terminate the collection of data if the controller shows stability
                # for a long time. This is a good thing.
                if steps > maximum_number_of_steps:
                    done_n[:] = True

        print 'Filled Buffer'

        for iteration in range(iteration_offset, iteration_offset + iterations + 1):
            print('{0} Iteration {1} {0}'.format('*'*10, iteration))
            networks[0].buffer.soft_reset()
            timesteps_in_iteration = 0

            if (iteration % update_freq == 0):
                saver.save(sess,os.path.join(logdir,'model-'+str(iteration)+'.cptk'))
                print "Saved Model. Timestep count: %s" % iteration

            total_number_of_steps_in_iteration = 0

            total_reward = np.array([0]*env.n_actors)

            while True:
                networks[0].buffer.games_played += 1
                if (((networks[0].buffer.games_played) % 10) == 0):
                    print 'Epoch: %s. Game number: %s' % (iteration, networks[0].buffer.games_played)
                obs = env.reset()
                epsilon_schedule.reset()

                # raw_observations = []
                # raw_observations.append(np.array(obs))


                animate_episode = ((networks[0].buffer.games_played-1)==0) and (iteration % update_freq == 0) and animate

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
                            time.sleep(.01)

                        monitor.add(rgb, iteration, networks[0].buffer.games_played)

                    length_alive[env.world.idxs_of_alive_snakes] += 1
                    # ob = get_data(np.array(raw_observations)[-2:])
                    last_obs = obs

                    # Control the exploration
                    acts = []
                    is_greedys = []
                    for i, network in enumerate(networks):
                        act, is_greedy = network.select_from_policy(np.array([[x.A for x in get_data(last_obs, i)]]), epsilon_schedule.value(network.epoch), eta_schedule.value(network.epoch)) 
                        acts += [str(act[0])]
                        is_greedys.append(is_greedy)
          
                    # Next step
                    obs, reward_n, done_n = env.step(acts)

                    total_reward += np.array(reward_n)
                    
                    total_number_of_steps_in_iteration += 1
                    steps += 1

                    for i in env.world.idxs_of_alive_snakes:
                        networks[i].store(np.array(get_data(last_obs, i)), # state
                                      np.array(acts[i]), # action
                                      np.array(reward_n[i]), #rewards
                                      np.array(get_data(obs, i)), #new state
                                      np.array(done_n[i]) #done
                                      )
                        if True: #is_greedys[i]:
                            networks[i].store_reservoir(np.array(get_data(last_obs, i)), # state
                                                        np.array(int(acts[i])))

                    # terminate the collection of data if the controller shows stability
                    # for a long time. This is a good thing.
                    if steps > maximum_number_of_steps:
                        done_n[:] = True

                if viewer:
                    viewer.close()

                if networks[0].buffer.games_played >= 1:
                    break

            # max: to cover all new steps added to buffer, min: to not overdo too much
            for i,network in enumerate(networks):
                for _ in range(min(max(maximum_number_of_steps/batch_size + 1,length_alive[i]/4), 12)): # learn every 4 frames, up to a total of 5 times.
                    if use_priority:
                        network.train_step(learning_rate_schedule, beta_schedule)
                    else:
                        network.train_step(learning_rate_schedule)

                for _ in range(2):
                    network.avg_policy_train_step(policy_learning_rate_schedule)

                monitor.make_gifs(iteration)
            
            
            for count, writer in enumerate(summary_writers):
                if count < (len(summary_writers) - 1):
                    summary = tf.Summary()
                    summary.value.add(tag='Average Reward', simple_value=(total_reward[count]))
                    summary.value.add(tag='Steps Taken', simple_value=(length_alive[count]))
                    writer.add_summary(summary, iteration)
                writer.flush()
                    
            

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--iterations', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--policy_batch_size', '-p_b', type=int, default=32)
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
    parser.add_argument('--use_priority', '-priority', action='store_true')
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
        buffer_size=args.buffer_size,
        use_priority=args.use_priority,
        policy_batch_size=args.policy_batch_size)

if __name__ == "__main__":
    main()