import numpy as np
import torch
from args import get_ddpg_args_test
#new
import cv2
from PIL import Image
from matplotlib import pyplot as plt#remove after debug
#-----

from ddpg import DDPG
from env import launch_env

from wrappers import ActionWrapper, ImgWrapper, NormalizeWrapper, ResizeWrapper

policy_name = "DDPG"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_ddpg_args_test()

file_name = "{}_{}".format(policy_name, args.seed)

env = launch_env()

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")

policy.load(file_name, directory="./pytorch_models")

cutoff = 256

with torch.no_grad():
    while True:
        obs = env.reset()
        
        
        #declare obs
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
        
        
        env.render()
        rewards = []
        steps = 0
        while True:
            action = policy.predict(np.array(obs))

            print(action)
            obsN, rew, done, misc = env.step(action)
            obsNEW = np.squeeze(obsN, axis=0)#remove unecessary third axis in this case first because it's transposed
            #get obs in
            obs[2,:,:]=obs[1,:,:]
            obs[1,:,:]=obs[0,:,:]
            obs[0,:,:]=obsNEW
            rewards.append(rew)
            env.render()
            steps += 1
            if done or steps >= cutoff:
                break
        print("mean episode reward:", np.mean(rewards))
