import argparse
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import numpy as np
import torch
import cv2
from run_test import *
from baselines.common.trex_utils import preprocess
import time


def generate_novice_demos(env, env_name, agent, model_dir, visualize):
    check_min, check_max, check_step = 50, 600, 50
    checkpoints = []
    if env_name == "enduro":
        check_min = 3100
        check_max = 3650
    elif env_name == "seaquest":
        check_min = 10
        check_max = 65
        check_step = 5
    for i in range(check_min, check_max + check_step, check_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(f'Checkpoints: {checkpoints}')

    demonstrations, learning_returns, learning_rewards = [], [], []
    for checkpoint in checkpoints:
        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj, actions, gt_rewards = [], [], []            
            r, steps, acc_reward, frameno = 0, 0, 0, 0

            ob = env.reset()
            if visualize:
                os.mkdir('images/' + str(checkpoint))
            
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                ob_processed = preprocess(ob, env_name)
                ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                traj.append(ob_processed)
                actions.append(action[0])
                frameno += 1
                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]

                if visualize:
                    cv2.imwrite(torch.from_numpy(ob_processed).permute(2, 0, 1).reshape(4*84, 84), 'images/' + str(checkpoint) + '/' + str(frameno) + '_action_' + str(action[0]) + '.png')
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print(f"Traj length {len(traj)}, demo length: {len(demonstrations)}")
            demonstrations.append([traj, actions])
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards


def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs, training_labels, times, actions = [], [], [], []
    num_demos = len(demonstrations)

    #add full trajs (for use on Enduro)
    for _ in range(num_trajs):
        ti, tj = 0, 0

        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti, tj = tuple(np.random.randint(num_demos, size=2))

        #create random partial trajs by finding random start frame and random skip frame
        si, sj = tuple(np.random.randint(6, size=2))
        step = np.random.randint(3,7)

        traj_i, traj_j = demonstrations[ti][0][si::step], demonstrations[tj][0][sj::step]  #slice(start,stop,step)
        traj_i_actions, traj_j_actions = demonstrations[ti][1][si::step], demonstrations[tj][1][sj::step] #skip everyother framestack to reduce size

        label = 0 if ti > tj else 1

        times.append((list(range(si, len(demonstrations[ti][0]), step)), list(range(sj, len(demonstrations[tj][0]), step))))
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        actions.append((traj_i_actions, traj_j_actions))
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti, tj = 0, 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti, tj = tuple(np.random.randint(num_demos, size=2))
            
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti][0]), len(demonstrations[tj][0]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj][0]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti][0]) - rand_length + 1)
        traj_i, traj_j = demonstrations[ti][0][ti_start:ti_start+rand_length:1], demonstrations[tj][0][tj_start:tj_start+rand_length:1] #skip everyother framestack to reduce size
        traj_i_actions, traj_j_actions = demonstrations[ti][1][ti_start:ti_start+rand_length:1], demonstrations[tj][1][tj_start:tj_start+rand_length:1] #skip everyother framestack to reduce size
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

        label = 0 if ti > tj else 1

        if len(traj_i) != len(list(range(ti_start, ti_start+rand_length, 1))):
            print("---------LENGTH MISMATCH!------")
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        times.append((list(range(ti_start, ti_start+rand_length, 1)), list(range(tj_start, tj_start+rand_length, 1))))
        actions.append((traj_i_actions, traj_j_actions))

    print("Maximum traj length", max_traj_length)
    return training_obs, training_labels, times, actions

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_trajs', default = 0, type=int, help="number of downsampled full trajectories")
    parser.add_argument('--num_snippets', default = 60000, type = int, help = "number of short subtrajectories to sample")
    parser.add_argument('--loss_fn', default='trex+ss', help="ss: selfsupervised, trex: only trex, trex+ss: both trex and selfsupervised")
    parser.add_argument('--visualize', default=False, action='store_true')

    args = parser.parse_args()
    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    print(f'Env type: {env_type}')

    #set seeds
    set_seed(args.seed)

    print("Training reward for", env_id)
    min_snippet_length, maximum_snippet_length = 50, 100 #min length of trajectory for training comparison

    if env_name == "enduro":
        print("only using full trajs for enduro")
        args.num_trajs = 10000
        args.num_snippets = 0

    # Hyperparameters
    num_iter = 2 if args.env_name == 'enduro' and args.loss_fn == 'trex+ss' else 1

    env = make_vec_env(env_id, 'atari', 1, args.seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })

    ACTION_DIMS = env.action_space.n
    print("Number of actions", ACTION_DIMS)

    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic=True)

    demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir, visualize=args.visualize)

    #sort the demonstrations according to ground truth reward to simulate ranked demos

    lenghts = [(len(d[0]), len(d[1])) for d in demonstrations]
    for demo_length, action_length in lenghts:
        assert demo_length == action_length
    maximum_snippet_length = min(np.min([d[0] for d in lenghts]), maximum_snippet_length)
    print(f'Max snippet length: {maximum_snippet_length}')
    print(f'Demo lengths: {[d[0] for d in lenghts]}')

    print(f'Length of learning returns: {len(learning_returns)}, demonstrations: {len(demonstrations)}')
    print(f'Learning returns and demonstrations: {[a[0] for a in zip(learning_returns, demonstrations)]}')
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(f'Sorted returns:{sorted_returns}')

    # Put all data in one array
    obs, labels, times, actions = create_training_data(demonstrations, args.num_trajs, args.num_snippets, min_snippet_length, maximum_snippet_length)
    print(f'Time: {round(time.time() - start, 4)}')

    folder = '../training_data'
    np.save(open(f'{folder}/training_obs.npy', 'wb'), np.array(obs, dtype=object))
    np.save(open(f'{folder}/training_labels.npy', 'wb'), np.array(labels, dtype=object))
    np.save(open(f'{folder}/training_actions.npy', 'wb'), np.array(actions, dtype=object))
    np.save(open(f'{folder}/training_times.npy', 'wb'), np.array(times, dtype=object))
    np.save(open(f'{folder}/demonstrations.npy', 'wb'), np.array(demonstrations, dtype=object))
    np.save(open(f'{folder}/sorted_returns.npy', 'wb'), np.array(sorted_returns, dtype=object))

    print('Saved training data successfully')
