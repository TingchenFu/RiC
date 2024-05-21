'''
This script is used to prepare the training dataset for RiC.
'''
from dataclasses import dataclass, field
from typing import Optional
import datasets
from datasets import load_dataset
from accelerate import Accelerator
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from my_reward_model import RewardModel
import os
import json
# define paths for two datasets
# hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
# summary_dataset_path = 'openai/summarize_from_feedback'


def template_function_hh(sample,chosen=True):
    if 'prompt' in sample and 'completion' in sample:
        return sample
    if chosen:
        text = sample['chosen']
    else:
        text = sample['rejected']
    split_text = text.split('\n\nAssistant:')
    sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
    sample['completion'] = split_text[-1].strip()
    return sample



def template_function_beaver(sample,criteria='better_response_id'):
    assert criteria in ['better_response_id', 'safer_response_id']
    chosen_id = int(sample[criteria])
    sample['completion'] = sample['response_{}'.format(chosen_id)]    

    return sample 


@dataclass
class ScriptArguments:
    reward_tokenizer_name_or_path: Optional[str] = field(default=None)
    reward_model_name_or_path: Optional[str] = field(default=None)
    input_file: Optional[str] = field(default=None, metadata={"help": "The input file to score and reward"})
    output_file: Optional[str] = field(default=None, metadata={"help": "The output file to save the rewards"})
    
    #reward_names:Optional[str] = field(default='helpfulharmless,humor') 


    def __post_init__(self):
        self.reward_model_name_or_path = self.reward_model_name_or_path.split(',')
        if self.reward_tokenizer_name_or_path is None:
            self.reward_tokenizer_name_or_path = self.reward_model_name_or_path
        else:
            self.reward_tokenizer_name_or_path = self.reward_tokenizer_name_or_path.split(',')
            assert len(self.reward_tokenizer_name_or_path) == len(self.reward_model_name_or_path), "The number of reward model paths and tokenizer paths should be the same."
        self.output_file = self.output_file.split(',')
        assert len(self.output_file) == len(self.reward_model_name_or_path)
    


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

gpu_id = Accelerator().local_process_index 
print("GPU ID: ", gpu_id)
reward_models = RewardModel(script_args.reward_model_name_or_path, script_args.reward_tokenizer_name_or_path, gpu_id)


raw_dataset = load_dataset('json', data_files = script_args.input_file.split(','))
# ATTENTION
raw_dataset = raw_dataset['train']
#raw_dataset = raw_dataset.select(range(100))
#raw_dataset = raw_dataset['train'].select(range(0,len(raw_dataset['train']),4))
#print("here is OK")
print(len(raw_dataset))
templated_dataset = raw_dataset.map(
    template_function_hh if 'HH' in script_args.input_file else template_function_beaver,
    batched=False,
    remove_columns= None,
)

# templated_dataset = templated_dataset.filter(lambda x: len(tokenizer.encode(x['prompt'])) <= 512 and len(tokenizer.encode(x['prompt'])) >= 8)

query_response = []
for i in range(len(templated_dataset)):
    query_response.append((templated_dataset[i]['prompt'], templated_dataset[i]['completion']))
reward_list = reward_models.get_multiple_reward_parallel(query_response)

fout = open(script_args.output_file[gpu_id], 'w')
json.dump(reward_list, fout)
