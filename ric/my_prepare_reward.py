'''
This script is used to prepare the training dataset for RiC.
'''
from dataclasses import dataclass, field
from typing import Optional
import datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk
from accelerate import Accelerator
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from utils import Instructions_n, load_main_tokenizer, Instructions_summary_n
from my_reward_model import RewardModel
import os
import json
# define paths for two datasets
# hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
# summary_dataset_path = 'openai/summarize_from_feedback'
tokenizer_name = '/home/futingchen/PLM/Llama-2-7b-hf'


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

@dataclass
class ScriptArguments:
    reward_names:Optional[str] = field(default='helpful,harmless,humor') 
    save_directory: Optional[str] = field(default='./data/HH/')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(reward_names)
reward_path_tokenizer_dict = {
    'harmless': ['/home/futingchen/PLM/gpt2large_harmless_reward'],
    'helpful': ['/home/futingchen/PLM/gpt2large_helpful_reward'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['/home/futingchen/PLM/distilbert_humor_reward'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
    

tokenizer = load_main_tokenizer(tokenizer_name)
gpu_id = Accelerator().local_process_index 
print("GPU ID: ", gpu_id)
reward_models = RewardModel(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
rm_tokenizers = reward_models.rm_tokenizers

raw_dataset = load_dataset('json', data_files=['/home/futingchen/MultiContrast/RiC/ric/dump/online1_Llama-2-7b-hf_HH_lora_bf16_bs4lr1e-5decay0.01constant_04111305_empty.jsonl'])
# ATTENTION
raw_dataset = raw_dataset['train']
#raw_dataset = raw_dataset.select(range(100))
#raw_dataset = raw_dataset['train'].select(range(0,len(raw_dataset['train']),4))
#print("here is OK")
print(len(raw_dataset))
templated_dataset = raw_dataset.map(
    template_function_hh,
    batched=False,
    remove_columns= None,
)

# templated_dataset = templated_dataset.filter(lambda x: len(tokenizer.encode(x['prompt'])) <= 512 and len(tokenizer.encode(x['prompt'])) >= 8)

query_response = []
for i in range(len(templated_dataset)):
    query_response.append((templated_dataset[i]['prompt'], templated_dataset[i]['completion']))
reward_list = reward_models.get_multiple_reward_parallel(query_response)

fout = open(os.path.join(script_args.save_directory, 'online2_reward_{}.json'.format(reward_names[gpu_id])), 'w')
json.dump(reward_list, fout)
