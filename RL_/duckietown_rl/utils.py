import random

import numpy as np
import torch
#new
import cv2
from PIL import Image
from matplotlib import pyplot as plt#remove after debug
#-----
def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done))

    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1, 1),
            "done": np.stack(dones).reshape(-1, 1),
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        obs=obs.transpose(1, 2, 0)
        obs = np.squeeze(obs, axis=2)#remove unecessary third axis
        image = Image.fromarray(obs)#convert to immage
        #plt.imshow(image)
        #plt.show()
        """print("efefefefefefe", obs.shape)
        image2 = (np.random.standard_normal([64,64,3]) * 255).astype(np.uint8)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
       
        #gray2 = np.expand_dims(gray2, axis=0)#add missing axis
        print("efefefefefefe", gray2.shape)
        
        image3 = (np.random.standard_normal([64,64,3]) * 255).astype(np.uint8)
        gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
        
        
     
        #gray3 = np.expand_dims(gray3, axis=0)#add missing axis
        print("efefefefefefe", gray3.shape)
        """

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
        print(obs.shape)
        done = False
        step = 0
        while not done and step < max_timesteps:
        
           
            
            action = policy.predict(np.array(obs))
            
            obsNEW, reward, done, _ = env.step(action)
            print("before",obsNEW.shape)
            obsNEW2 = np.squeeze(obsNEW, axis=0)#remove unecessary third axis in this case first
            print("after", obsNEW2.shape)
            print("obs",obs[2,:,:].shape)
            #obs is 2 dimentional
            obs[2,:,:]=obs[1,:,:]
            obs[1,:,:]=obs[0,:,:]
            obs[0,:,:]=obsNEW2
            #obs3=obs2
            #obs2=obs1
            #obs1 =obsNEW
            
            avg_reward += reward
            step += 1
    
    
    """obs=obs.transpose(1, 2, 0)
    obss = obs[:,:,2]#2 dim
    image = Image.fromarray(obss)#convert to immage
    plt.imshow(image)
    plt.show()"""
    avg_reward /= eval_episodes

    return avg_reward
    
    
    
    
