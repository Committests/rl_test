import pickle
import random
import resource
import gym_duckietown
import numpy as np
import torch
import gym
import os
#new
import cv2
from PIL import Image
from matplotlib import pyplot as plt#remove after debug
#-----

from args import get_ddpg_args_train
from ddpg import DDPG
from utils import seed, evaluate_policy, ReplayBuffer
from wrappers import (
    NormalizeWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
    SteeringToWheelVelWrapper,
)
from env import launch_env
from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise#new riza
policy_name = "DDPG"

print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_ddpg_args_train()

if args.log_file != None:
    print('You asked for a log file. "Tee-ing" print to also print to file "' + args.log_file + '" now...')

    import subprocess, os, sys

    tee = subprocess.Popen(["tee", args.log_file], stdin=subprocess.PIPE)
    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


file_name = "{}_{}".format(
    policy_name,
    str(args.seed),
)

if not os.path.exists("./results"):
    os.makedirs("./results")
if args.save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

env = launch_env()

# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")

replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

# Evaluate untrained policy
evaluations = [evaluate_policy(env, policy)]

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0
# Create OrnsteinUhlenbeckActionNoise instance new riza
action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2), sigma=np.ones(2)*0.2, theta=0.7)
while total_timesteps < args.max_timesteps:

    if done:
        print(f"Done @ {total_timesteps}")

        if total_timesteps != 0:
            print("Replay buffer length is ", len(replay_buffer.storage))
            print(
                ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                % (total_timesteps, episode_num, episode_timesteps, episode_reward)
            )
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy))

            if args.save_models:
                policy.save(file_name, directory="./pytorch_models")
            np.savez("./results/{}.npz".format(file_name), evaluations)

        # Reset environment with initial 3 frames
        env_counter += 1
        obs = env.reset()
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
       
        obs=obs.transpose(1, 2, 0)
        obs = np.squeeze(obs, axis=2)#remove unecessary third axis
        image = Image.fromarray(obs)#convert to immage
        #plt.imshow(image)
        #plt.show()

        """Create a blank image that has three channels
        and the same number of pixels as your original input"""
        needed_multi_channel_img = np.zeros((64, 64, 3))

        """Add the channels to the needed image one by one"""
        needed_multi_channel_img [:,:,0] = image
        needed_multi_channel_img [:,:,1] = image
        needed_multi_channel_img [:,:,2] = image

        #obs = cv2.merge([gray4,gray2,gray3])
        print(needed_multi_channel_img.shape)
        #plt.imshow(needed_multi_channel_img[:,:,1])
        #plt.show()
        obs = np.array(needed_multi_channel_img)#convert back to array
        obs=obs.transpose(2, 0, 1)#transpose back
        
#---------------------add old frames to obs
        
        
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        
        action_noise.reset()#new riza

    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.predict(np.array(obs))
        action = action + action_noise() * (1 - total_timesteps / args.max_timesteps) * [1 if episode_num % 2 == 0 else 0.5][0]#rza new
        '''if args.expl_noise != 0:
            action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high
            )'''

    # Perform action
    new_obs, reward, done, _ = env.step(action)
    #---------------add obs in new obs and previous one
    obsNEW = np.squeeze(new_obs, axis=0)#remove unecessary third axis in this case first because it's transposed

    
    if (
        action[0] < 0.04#0.01 is good  (it's the velocity because action[vel, angle])(0.01)  204: 0.001 301: 0.01 302: 0.001 \\ 303: 005 304: 007 205 003
    ):  # Penalise slow actions: helps the bot to figure out that going straight > turning in circles
        reward = -10 # good(4) - 10 is too much, 3 is too low, (204: 3.5.. 300: 6) 301: 4 302:5 da ora in poi tutti 4  2 0.007 305 306 0.001 4 206....... 307 -3 007, 308 -5 008  207 -4 008.......... 01 1000 208, 4 002 309, 308 001 10
    '''if (
        action[0] == 0#0.001 (it's the velocity because action[vel, angle])(0.01)
    ):  # Penalise slow actions: helps the bot to figure out that going straight > turning in circles
        reward = -70#probably to high(4)'''
        
    #put collision with white line

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
    episode_reward += reward
    
    #mooved the change here for the replay buffer
    oldb=obs
    obs[2,:,:]=obs[1,:,:]
    obs[1,:,:]=obs[0,:,:]
    obs[0,:,:]=obsNEW
    newb=obs

    # Store data in replay buffer
    replay_buffer.add(oldb, newb, action, reward, done_bool)
    # approximate_size(replay_buffer)   #TODO rm

    #obs = new_obs but for 3 gray images it's up 169
 
    
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(env, policy))

if args.save_models:
    policy.save(file_name, directory="./pytorch_models")
np.savez("./results/{}.npz".format(file_name), evaluations)
