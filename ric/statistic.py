import os
import json
import numpy as np
helpful_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/reward_helpful.json'))
harmless_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/reward_harmless.json'))
humor_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/reward_humor.json'))

helpful_mean = np.mean(helpful_reward)
helpful_std = np.std(helpful_reward)
normed_reward = [(x-helpful_mean)/helpful_std for x in helpful_reward]
json.dump(normed_reward, open('/home/futingchen/MultiContrast/RiC/ric/data/HH/norm_reward_helpful.json', 'w'))

harmless_mean = np.mean(harmless_reward)
harmless_std = np.std(harmless_reward)
normed_reward = [(x-harmless_mean)/harmless_std for x in harmless_reward]
json.dump(normed_reward, open('/home/futingchen/MultiContrast/RiC/ric/data/HH/norm_reward_harmless.json', 'w'))

humor_mean = np.mean(humor_reward)
humor_std = np.std(humor_reward)
normed_reward = [(x-humor_mean)/humor_std for x in humor_reward]
json.dump(normed_reward, open('/home/futingchen/MultiContrast/RiC/ric/data/HH/norm_reward_humor.json', 'w'))
 
reward_mean = [helpful_mean, harmless_mean, humor_mean]
reward_std = [helpful_std, harmless_std, humor_std]
np.save('/home/futingchen/MultiContrast/RiC/ric/data/HH/reward_statistic.json', np.array([reward_mean, reward_std]).reshape(2, -1).T)