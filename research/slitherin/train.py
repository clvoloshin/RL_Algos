import pdb
import time
import numpy as np
import os
import tensorflow as tf
import gym
import gym_snake
import scipy.sparse as sparse
from copy import deepcopy
from gym.envs.classic_control import rendering
from history import History
from action_policy_network import Policy
from mcts.mcts import *
from mcts.graph import Node



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
   


    mcts = MCTS(UCB(np.sqrt(2)), backup, len(env.world.action_space), env.n_actors)



    network = Policy(env)

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
        history = History(5000)

        for iteration in range(iteration_offset, iteration_offset + iterations):
            print('{0} Iteration {1} {0}'.format('*'*10, iteration))

            timesteps_in_iteration = 0
            history.reset()


            if (iteration % 10 == 0):
                saver.save(sess,os.path.join(logdir,'model-'+str(iteration)+'.cptk'))
                print "Saved Model. Timestep count: %s" % iteration


                # print 'Evalauting new model against previous best'
                # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
                # pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

                # print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
                # if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                #     print('REJECTING NEW MODEL')
                #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                # else:
                #     print('ACCEPTING NEW MODEL')
                #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')   


            game_num = 0
            while True:
                game_num += 1
                print 'Game number: %s' % game_num
                obs = env.reset()


                raw_observations, observations, actions, rewards, pis = [], [], [], [], []

                animate_episode = (history.get_total_timesteps()==0) and (iteration % 10 == 0) and animate

                done_n = np.array([False]*env.n_actors)
                steps = 0

                # Runs policy, collects observations and rewards
                viewer = None

                length_alive = np.array([0] * env.n_actors)
                roots = [None] * env.n_actors
                while not done_n.all():
                    if animate_episode:
                        if not viewer:
                            viewer = rendering.SimpleImageViewer()
                        rgb = env.render('rgb_array')
                        scaler = 10
                        upscaled=repeat_upsample(rgb,scaler,scaler)
                        viewer.imshow(upscaled)
                        time.sleep(.000001)

                    length_alive[env.world.idxs_of_alive_snakes] += 1

                    raw_observations.append(np.array(obs))
                    ob = get_data(np.array(raw_observations)[-2:])
                    
                    P, v = sess.run([network.softmax_policy, network.pred_Q], feed_dict={network.obs: np.array([[y.A for y in x] for x in ob]), 
                                                            network.training_flag: False})
                    
                    pi = []
                    acts = []
                    for actor in np.arange(env.n_actors):
                        if roots[actor] is None:
                            roots[actor] = Node(None, deepcopy(env.world), 1, env.n_actors, len(env.world.action_space), actor, Ps=P[actor])
                        pi.append(mcts.run(sess, network, deepcopy(env.world), roots[actor], mcts_iterations))
                        acts.append(np.random.choice(len(env.world.action_space), p = pi[-1]))
                        roots[actor] = roots[actor].children[str(acts[-1])]
                        roots[actor].parent = None

                    pis.append(pi)
                    actions.append([str(x) for x in acts])

                    observations.append(ob)

                                      
                    



                    # actions.append([str(x) for x in action])
                    
                    # Next step
                    obs, reward_n, done_n = env.step(actions[-1])
                    rewards.append(reward_n)
                    steps += 1

                    # terminate the collection of data if the controller shows stability
                    # for a long time. This is a good thing.
                    if steps > maximum_number_of_steps:
                        done_n[:] = True

                history.append(obs=np.array(observations), 
                               actions=np.array(actions), 
                               rewards=np.array(rewards),
                               policies=np.array(pis),
                               lengths=np.array(length_alive))
                
                if history.games_played > games_played_per_epoch:
                    break


            # pdb.set_trace()
            for batch in range(num_batches):
                print batch
                obs_batch, act_batch, rew_batch, policies_batch = history.sample(batch_size, discount)

                sess.run(network.train, feed_dict={network.learning_rate: 3e-4, 
                                                   network.training_flag: True,
                                                   network.obs: np.array([[y.A for y in x] for x in obs_batch]), 
                                                   network.target_policy: policies_batch, 
                                                   network.target_Q: rew_batch })

            history.soft_reset()


            # Log diagnostics
            # returns = history.get_total_reward_per_sequence()
            # ep_lengths = history.get_timesteps()


            # summary = tf.Summary()
            # summary.value.add(tag='Average Return', simple_value=np.mean(returns))
            # summary.value.add(tag='Std Return', simple_value=np.std(returns))
            # summary.value.add(tag='Max Return', simple_value=np.max(returns))
            # summary.value.add(tag='Min Return', simple_value=np.min(returns))
            # summary.value.add(tag='Episode Length Mean', simple_value=np.mean(ep_lengths))
            # summary.value.add(tag='Episode Length Std', simple_value=np.std(ep_lengths))
            # summary.value.add(tag='Timesteps in Batch', simple_value=history.get_total_timesteps())
            # summary.value.add(tag='Total Timesteps', simple_value=total_timesteps)
            # summary_writer.add_summary(summary, iteration)
            # summary_writer.flush()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--iterations', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--num_batches', '-n_batch', type=int, default=10) # num batch/epoch
    parser.add_argument('--sequence_length', '-seq', type=int, default=200)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-num_exp', type=int, default=1)
    parser.add_argument('--games_played_per_epoch', '-gpe', type=int, default=10)
    parser.add_argument('--mcts_iterations', '-mcts', type=int, default=1500)
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
        mcts_iterations=args.mcts_iterations)

if __name__ == "__main__":
    main()