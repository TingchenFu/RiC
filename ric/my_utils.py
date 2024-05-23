from datasets import load_dataset,concatenate_datasets
import json
import numpy as np
from pathlib import Path 
import os
file = Path(__file__).resolve()
parent, root, home = file.parent, file.parents[1], file.parents[2]



def template_function_hh(sample,chosen=True):
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


def get_offline_train_HH():
    raw_dataset = load_dataset('json', data_files=[os.path.join(home, 'data/HH/harmless_base_train.jsonl'), os.path.join(home,'data/HH/helpful_base_train.jsonl'), os.path.join(home, 'data/HH/helpful_online_train.jsonl'), os.path.join(home, 'data/HH/helpful_sampled_train.jsonl')],split='train')

    offline_helpful_reward = json.load(open( os.path.join(home, 'RiC/ric/data/HH/norm_reward_harmless.json') ))
    offline_harmless_reward = json.load(open( os.path.join(home, 'RiC/ric/data/HH/norm_reward_helpful.json')))
    offline_honor_reward = json.load(open( os.path.join(home, 'RiC/ric/data/HH/norm_reward_humor.json')))

    raw_dataset = raw_dataset.add_column('helpful_reward', offline_helpful_reward)
    raw_dataset = raw_dataset.add_column('harmless_reward', offline_harmless_reward)
    raw_dataset = raw_dataset.add_column('humor_reward', offline_honor_reward)

    print(raw_dataset.column_names)
    print(raw_dataset[0])

    raw_dataset.to_json( os.path.join(home,'RiC/ric/data/HH/train.jsonl'), orient='records')


def get_offline_train_beaver():
    raw_dataset = load_dataset('json', data_files=[os.path.join(home,'data/beaver_train.jsonl')],split='train')

    offline_helpful_reward = json.load(open(os.path.join(home, 'RiC/ric/data/beaver/offline_reward_helpful.json')))
    offline_harmless_reward = json.load(open(os.path.join(home, 'RiC/ric/data/beaver/offline_reward_harmless.json')))

    raw_dataset = raw_dataset.add_column('helpful_reward', offline_helpful_reward)
    raw_dataset = raw_dataset.add_column('harmless_reward', offline_harmless_reward)

    print(raw_dataset.column_names)
    print(raw_dataset[0])

    raw_dataset.to_json( os.path.join(home, 'RiC/ric/data/beaver/offline_train.jsonl'), orient='records')



def get_online1_train_HH():
    offline_dataset = load_dataset('json', data_files=['/home/tingchen_fu/MultiContrast/RiC/ric/data/HH/offline_train.jsonl'],split='train')


    thresholds = [np.quantile(offline_dataset['helpful_reward'], 0.7),
                np.quantile(offline_dataset['harmless_reward'], 0.7),
                np.quantile(offline_dataset['humor_reward'], 0.7)]
    offline_dataset = offline_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) + int(x['humor_reward'] >= thresholds[2])>=2)


    offline_dataset = offline_dataset.map(template_function_hh, batched=False, remove_columns=['chosen','rejected'])
    offline_dataset = offline_dataset.select(range(10000))
                                            

    online1_dataset = load_dataset('json', data_files=['/home/tingchen_fu/MultiContrast/RiC/ric/dump/offline_phi-2_HH_lora_bf16_bs4lr1.41e-4decay0.01linear_04112314_empty.jsonl'],split='train')
    online1_helpful_reward = json.load(open('/home/tingchen_fu/MultiContrast/RiC/ric/data/HH/online1_reward_helpful.json'))
    online1_harmless_reward = json.load(open('/home/tingchen_fu/MultiContrast/RiC/ric/data/HH/online1_reward_harmless.json'))
    online1_humor_reward = json.load(open('/home/tingchen_fu/MultiContrast/RiC/ric/data/HH/online1_reward_humor.json'))

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


    concatenate_datasets([offline_dataset,online1_dataset]).to_json('/home/tingchen_fu/MultiContrast/RiC/ric/data/HH/online1_train.jsonl', orient='records')





# RiC/ric/data/beaver/offline_train.jsonl

def get_online1_train_beaver():
    offline_dataset = load_dataset('json', data_files = os.path.join(home, 'RiC/ric/data/beaver/offline_train.jsonl'),split='train')


    thresholds = [np.quantile(offline_dataset['helpful_reward'], 0.7),
                np.quantile(offline_dataset['harmless_reward'], 0.7)]
    offline_dataset = offline_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) >=1)


    offline_dataset = offline_dataset.map(template_function_beaver, batched=False, remove_columns=None)
    offline_dataset = offline_dataset.select(range(10000))
                                            

    online1_dataset = load_dataset('json', data_files=[os.path.join(home, 'RiC/ric/dump_beaver/offline_phi-2_beaver_lora_bf16_bs4lr1.41e-4decay0.01linear_05221853_empty.jsonl')],split='train')
    online1_helpful_reward = json.load(open(os.path.join(home, 'RiC/ric/data/beaver/online1_reward_helpful.json')))
    online1_harmless_reward = json.load(open(os.path.join(home, 'RiC/ric/data/beaver/online1_reward_harmless.json')))

    online1_dataset = online1_dataset.add_column('helpful_reward', online1_helpful_reward)
    online1_dataset = online1_dataset.add_column('harmless_reward', online1_harmless_reward)


    thresholds = [np.quantile(online1_dataset['helpful_reward'], 0.7),
                np.quantile(online1_dataset['harmless_reward'], 0.7)]
    online1_dataset = online1_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) >=1)

    print(len(offline_dataset))
    print(len(online1_dataset))

    print(offline_dataset.column_names)
    print(online1_dataset.column_names)


    concatenate_datasets([offline_dataset,online1_dataset]).to_json(os.path.join(home, 'RiC/ric/data/beaver/online1_train.jsonl'), orient='records')


def get_online2_train_HH():
    offline_dataset = load_dataset('json', data_files=[os.path.join(home, 'RiC/ric/data/HH/offline_train.jsonl')],split='train')


    thresholds = [np.quantile(offline_dataset['helpful_reward'], 0.7),
                np.quantile(offline_dataset['harmless_reward'], 0.7),
                np.quantile(offline_dataset['humor_reward'], 0.7)]
    offline_dataset = offline_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) + int(x['humor_reward'] >= thresholds[2])>=2)


    offline_dataset = offline_dataset.map(template_function_hh, batched=False, remove_columns=['chosen','rejected'])
    offline_dataset = offline_dataset.select(range(10000))
                                            

    online1_dataset = load_dataset('json', data_files=[os.path.join(home, 'RiC/ric/dump/offline_phi-2_HH_lora_bf16_bs4lr1.41e-4decay0.01linear_04112314_empty.jsonl')],split='train')
    online1_helpful_reward = json.load(open(os.path.join(home, '/RiC/ric/data/HH/online1_reward_helpful.json')))
    online1_harmless_reward = json.load(open(os.path.join(home, 'RiC/ric/data/HH/online1_reward_harmless.json')))
    online1_humor_reward = json.load(open(os.path.join(home,'RiC/ric/data/HH/online1_reward_humor.json')))

    online1_dataset = online1_dataset.add_column('helpful_reward', online1_helpful_reward)
    online1_dataset = online1_dataset.add_column('harmless_reward', online1_harmless_reward)
    online1_dataset = online1_dataset.add_column('humor_reward', online1_humor_reward)


    thresholds = [np.quantile(online1_dataset['helpful_reward'], 0.7),
                np.quantile(online1_dataset['harmless_reward'], 0.7),
                np.quantile(online1_dataset['humor_reward'], 0.7)]
    online1_dataset = online1_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) + int(x['humor_reward'] >= thresholds[2])>=2)


    online2_dataset = load_dataset('json', data_files=[os.path.join(home, 'RiC/ric/dump/online1_Llama-2-7b-hf_HH_lora_bf16_bs4lr1e-5decay0.01constant_04121428_empty.jsonl')],split='train')
    online2_helpful_reward = json.load(open( os.path.join(home, 'RiC/ric/data/HH/online2_reward_helpful.json')))
    online2_harmless_reward = json.load(open( os.path.join(home, 'RiC/ric/data/HH/online2_reward_harmless.json')))
    online2_humor_reward = json.load(open(os.path.join(home,'RiC/ric/data/HH/online2_reward_humor.json')))

    online2_dataset = online2_dataset.add_column('helpful_reward', online2_helpful_reward)
    online2_dataset = online2_dataset.add_column('harmless_reward', online2_harmless_reward)
    online2_dataset = online2_dataset.add_column('humor_reward', online2_humor_reward)


    thresholds = [np.quantile(online2_dataset['helpful_reward'], 0.7),
                np.quantile(online2_dataset['harmless_reward'], 0.7),
                np.quantile(online2_dataset['humor_reward'], 0.7)]
    online2_dataset = online2_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) + int(x['humor_reward'] >= thresholds[2])>=2)

    print(len(offline_dataset))
    print(len(online1_dataset))
    print(len(online2_dataset))

    print(offline_dataset.column_names)
    print(online1_dataset.column_names)
    print(online2_dataset.column_names)

    concatenate_datasets([offline_dataset,online1_dataset, online2_dataset]).to_json( os.path.join(home,'RiC/ric/data/HH/online2_train.jsonl'), orient='records')









def get_online2_train_beaver():
    offline_dataset = load_dataset('json', data_files=[os.path.join(home, 'RiC/ric/data/beaver/offline_train.jsonl')],split='train')


    thresholds = [np.quantile(offline_dataset['helpful_reward'], 0.7),
                np.quantile(offline_dataset['harmless_reward'], 0.7),
                ]
    offline_dataset = offline_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) >= 1)


    offline_dataset = offline_dataset.map(template_function_beaver, batched=False, remove_columns=None)
    offline_dataset = offline_dataset.select(range(10000))


    online1_dataset = load_dataset('json', data_files=[os.path.join(home, 'RiC/ric/data/beaver/online1_train.jsonl')],split='train')


    thresholds = [np.quantile(online1_dataset['helpful_reward'], 0.7),
                np.quantile(online1_dataset['harmless_reward'], 0.7),
                ]
    online1_dataset = online1_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) >=1 )


    online2_dataset = load_dataset('json', data_files=[os.path.join(home, 'RiC/ric/dump_beaver/online1_phi-2_beaver_lora_bf16_bs4lr1e-5decay0.01constant_05222124_empty.jsonl')],split='train')
    online2_helpful_reward = json.load(open( os.path.join(home, 'RiC/ric/data/beaver/online2_reward_helpful.json')))
    online2_harmless_reward = json.load(open( os.path.join(home, 'RiC/ric/data/beaver/online2_reward_harmless.json')))

    online2_dataset = online2_dataset.add_column('helpful_reward', online2_helpful_reward)
    online2_dataset = online2_dataset.add_column('harmless_reward', online2_harmless_reward)


    thresholds = [np.quantile(online2_dataset['helpful_reward'], 0.7),
                np.quantile(online2_dataset['harmless_reward'], 0.7),
                ]
    online2_dataset = online2_dataset.filter(lambda x: int(x['helpful_reward'] >= thresholds[0]) + int(x['harmless_reward'] >= thresholds[1]) >=1)

    print(len(offline_dataset))
    print(len(online1_dataset))
    print(len(online2_dataset))

    print(offline_dataset.column_names)
    print(online1_dataset.column_names)
    print(online2_dataset.column_names)

    concatenate_datasets([offline_dataset,online1_dataset, online2_dataset]).to_json( os.path.join(home,'RiC/ric/data/beaver/online2_train.jsonl'), orient='records')






get_online2_train_beaver()