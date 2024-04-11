from datasets import load_dataset,concatenate_datasets
import json
import numpy as np


def template_function_hh(sample,chosen=True):
    if chosen:
        text = sample['chosen']
    else:
        text = sample['rejected']
    split_text = text.split('\n\nAssistant:')
    sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
    sample['completion'] = split_text[-1].strip()
    return sample



def normalize():
    raw_dataset = load_dataset('json', data_files=['/home/futingchen/MultiContrast/data/HH/harmless_base_train.jsonl','/home/futingchen/MultiContrast/data/HH/helpful_base_train.jsonl','/home/futingchen/MultiContrast/data/HH/helpful_online_train.jsonl', '/home/futingchen/MultiContrast/data/HH/helpful_sampled_train.jsonl'],split='train')

    norm_helpful_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/norm_reward_harmless.json'))
    norm_harmless_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/norm_reward_helpful.json'))
    norm_honor_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/norm_reward_humor.json'))

    raw_dataset = raw_dataset.add_column('helpful_reward', norm_helpful_reward)
    raw_dataset = raw_dataset.add_column('harmless_reward', norm_harmless_reward)
    raw_dataset = raw_dataset.add_column('humor_reward', norm_honor_reward)

    print(raw_dataset.column_names)
    print(raw_dataset[0])

    raw_dataset.to_json('/home/futingchen/MultiContrast/RiC/ric/data/HH/train.jsonl', orient='records')



def get_online1_train():
    offline_dataset = load_dataset('json', data_files=['/home/futingchen/MultiContrast/RiC/ric/data/HH/offline_train.jsonl'],split='train')


    thresholds = [np.quantile(offline_dataset['helpful_reward'], 0.7),
                np.quantile(offline_dataset['harmless_reward'], 0.7),
                np.quantile(offline_dataset['humor_reward'], 0.7)]
    offline_dataset = offline_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) + int(x['humor_reward'] >= thresholds[2])>=2)


    offline_dataset = offline_dataset.map(template_function_hh, batched=False, remove_columns=['chosen','rejected'])
    offline_dataset = offline_dataset.select(range(10000))
                                            

    online1_dataset = load_dataset('json', data_files=['/home/futingchen/MultiContrast/RiC/ric/dump/offline_Llama-2-7b-hf_HH_lora_bf16_bs4lr1.41e-4decay0.01linear_04102114_empty.jsonl'],split='train')
    online1_helpful_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/online1_reward_helpful.json'))
    online1_harmless_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/online1_reward_harmless.json'))
    online1_humor_reward = json.load(open('/home/futingchen/MultiContrast/RiC/ric/data/HH/online1_reward_humor.json'))

    online1_dataset = online1_dataset.add_column('helpful_reward', online1_helpful_reward)
    online1_dataset = online1_dataset.add_column('harmless_reward', online1_harmless_reward)
    online1_dataset = online1_dataset.add_column('humor_reward', online1_humor_reward)


    thresholds = [np.quantile(online1_dataset['helpful_reward'], 0.7),
                np.quantile(online1_dataset['harmless_reward'], 0.7),
                np.quantile(online1_dataset['humor_reward'], 0.7)]
    online1_dataset = online1_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) + int(x['humor_reward'] >= thresholds[2])>=2)

    print(len(offline_dataset))
    print(len(online1_dataset))

    print(offline_dataset.column_names)
    print(online1_dataset.column_names)


    concatenate_datasets([offline_dataset,online1_dataset]).to_json('/home/futingchen/MultiContrast/RiC/ric/data/HH/online1_train.jsonl', orient='records')


get_online1_train()