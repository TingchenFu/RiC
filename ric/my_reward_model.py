from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
#from utils import load_reward_model, get_rewards
from tqdm import tqdm



# def load_reward_model(reward_peft_path):
#     num_labels = 2 if ('humor' in reward_peft_path or 'faithful' in reward_peft_path) else 1
#     reward_model = AutoModelForSequenceClassification.from_pretrained(
#                     reward_peft_path,
#                     num_labels=num_labels, torch_dtype=torch.bfloat16,
#                     #device_map='auto',
#                     )
#     # the reward model are not PEFT trained
#     # if check_lora_in_model_path(reward_model, reward_peft_path):
#     #     reward_model = PeftModel.from_pretrained(reward_model, reward_peft_path)
#     # if hasattr(reward_model, 'merge_and_unload'):
#     #     reward_model = reward_model.merge_and_unload() # merge lora weights
#     return reward_model



class RewardModel():
    @classmethod
    def load_reward_model(reward_peft_path):
        num_labels = 2 if ('humor' in reward_peft_path or 'faithful' in reward_peft_path) else 1
        reward_model = AutoModelForSequenceClassification.from_pretrained(
                        reward_peft_path,
                        num_labels=num_labels, torch_dtype=torch.bfloat16,
                        #device_map='auto',
                        )
        return reward_model
    
    def __init__(self, reward_model_path_list, rm_tokenizer_path_list, gpu_id=None, reward_stats_path=None):
        assert len(reward_model_path_list) == len(rm_tokenizer_path_list)
        self.reward_model_path_list = reward_model_path_list
        self.rm_tokenizer_path_list = rm_tokenizer_path_list
        self.num_rewards = len(reward_model_path_list)
        self.reward_stats = np.load(reward_stats_path) if reward_stats_path is not None else None
        self.reward_models = []
        self.rm_tokenizers = []
        self.gpu_id = gpu_id

        print('Loading {} reward models '.format(self.num_rewards))
        for i in range(self.num_rewards):
            self.reward_models.append(AutoModelForSequenceClassification.from_pretrained(
                        self.reward_model_path_list[i],
                        num_labels =  2 if ('humor' in self.reward_model_path_list[i] or 'faithful' in self.reward_model_path_list[i]) else 1,
                        torch_dtype=torch.bfloat16,
                        device_map=gpu_id,
                    )
            )
            self.rm_tokenizers.append(AutoTokenizer.from_pretrained(self.rm_tokenizer_path_list[i]))
        assert len(self.reward_models) ==  self.num_rewards == len(self.rm_tokenizers)
        #print(self.reward_model_path_list[2])




        # the reward model are not PEFT trained
        # if check_lora_in_model_path(reward_model, reward_peft_path):
        #     reward_model = PeftModel.from_pretrained(reward_model, reward_peft_path)
        # if hasattr(reward_model, 'merge_and_unload'):
        #     reward_model = reward_model.merge_and_unload() # merge lora weights
        
    def get_single_reward(self, reward_model, text_for_single_reward, reward_mean_std=None, sub_position=0):
        rewards = []
        print('log: reward model forwarding ...')
        with torch.no_grad():
            pbar = tqdm(total=len(text_for_single_reward))
            for inputs in text_for_single_reward:
                if sub_position != 0: # for multiple output of a reward model
                    rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0][sub_position])
                else:
                    rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0])
                pbar.update(1)
        
        if reward_mean_std is None:
            rewards = [np.round(r.cpu().detach().item(), 1) for r in rewards]
        else:
            mean_reward, std_reward = reward_mean_std
            rewards = [np.round((r.cpu().detach().item() - mean_reward) / std_reward, 1) for r in rewards]
        return rewards


    def get_multiple_reward(self, queries_responses, summary_fun=None):
        '''
        queries_responses: list of tuples of query and response
        '''
        text_for_multiple_reward = [] # a list of list of encoded texts for each reward model
        #i = min(self.gpu_id,self.num_rewards-1)
        for i in range(self.num_rewards):

            if i >= 1 and self.rm_tokenizer_path_list[i] == self.rm_tokenizer_path_list[i-1]: # two repition of the same rewards
                text_for_multiple_reward.append(text_for_multiple_reward[-1])
            if 'faithful' in self.reward_model_path_list[i]:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](text=r, text_pair=summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                text_for_multiple_reward.append(temp_encoded_texts)
            elif 'summary' in self.reward_model_path_list[i] or 'summarization' in self.reward_model_path_list[i]: # reverse prompt and response
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r + " " + self.rm_tokenizers[i].bos_token + " " + summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                text_for_multiple_reward.append(temp_encoded_texts)
            elif 'humor' in self.reward_model_path_list[i]: # use only response
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                text_for_multiple_reward.append(temp_encoded_texts)
            else:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](q, r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                text_for_multiple_reward.append(temp_encoded_texts)

        # normalize reward
        rewards = []
        for i in range(self.num_rewards):

            if self.reward_stats is not None:
                if type(self.reward_stats) == list or len(self.reward_stats) == 2 * self.num_rewards:
                    reward_mean_std = (self.reward_stats[2*i], self.reward_stats[2*i+1])
                else:
                    reward_mean_std = self.reward_stats[i]
            else:
                reward_mean_std = None


            if 'humor' in self.reward_model_path_list[i] or 'faithful' in self.reward_model_path_list[i]:
                single_reward = self.get_single_reward(self.reward_models[i], text_for_multiple_reward[i], reward_mean_std=reward_mean_std, sub_position=1)
            else:
                single_reward = self.get_single_reward(self.reward_models[i], text_for_multiple_reward[i], reward_mean_std=reward_mean_std)
            rewards.append(single_reward)


        return rewards


    def get_multiple_reward_parallel(self, queries_responses, summary_fun=None):
        '''
        queries_responses: list of tuples of query and response
        '''
        text_for_multiple_reward_parallel = None # a list of list of encoded texts for each reward model
        i = self.gpu_id
        if self.gpu_id >= self.num_rewards:
            exit()
        i = min(self.gpu_id,self.num_rewards-1)

        # if i >= 1 and self.rm_tokenizer_path_list[i] == self.rm_tokenizer_path_list[i-1]: # two repition of the same rewards
        #     text_for_multiple_reward.append(text_for_multiple_reward[-1])
        if 'faithful' in self.reward_model_path_list[i]:
            max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
            temp_encoded_texts = [self.rm_tokenizers[i](text=r, text_pair=summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
            text_for_multiple_reward_parallel = temp_encoded_texts
        elif 'summary' in self.reward_model_path_list[i] or 'summarization' in self.reward_model_path_list[i]: # reverse prompt and response
            max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
            temp_encoded_texts = [self.rm_tokenizers[i](r + " " + self.rm_tokenizers[i].bos_token + " " + summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
            text_for_multiple_reward_parallel = temp_encoded_texts
        elif 'humor' in self.reward_model_path_list[i]: # use only response
            max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
            temp_encoded_texts = [self.rm_tokenizers[i](r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
            text_for_multiple_reward_parallel = temp_encoded_texts
        else:
            max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
            temp_encoded_texts = [self.rm_tokenizers[i](q, r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
            text_for_multiple_reward_parallel = temp_encoded_texts

        # normalize reward
        if self.reward_stats is not None:
            if type(self.reward_stats) == list or len(self.reward_stats) == 2 * self.num_rewards:
                reward_mean_std = (self.reward_stats[2*i], self.reward_stats[2*i+1])
            else:
                reward_mean_std = self.reward_stats[i]
        else:
            reward_mean_std = None
        if 'humor' in self.reward_model_path_list[i] or 'faithful' in self.reward_model_path_list[i]:
            reward = self.get_single_reward(self.reward_models[i], text_for_multiple_reward_parallel, reward_mean_std=reward_mean_std, sub_position=1)
        else:
            reward = self.get_single_reward(self.reward_models[i], text_for_multiple_reward_parallel, reward_mean_std=reward_mean_std)
        return reward