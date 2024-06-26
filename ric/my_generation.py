import argparse
from datasets import load_dataset   
from transformers import AutoModelForCausalLM
from peft import PeftModel
import os
import torch
from vllm import LLM, SamplingParams
import json
from trl import set_seed
import sys
import numpy as np
set_seed(0)
from pathlib import Path 
file = Path(__file__).resolve()
parent, root, home = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(root))



def template_function_hh(sample,chosen=True):
    if chosen:
        text = sample['chosen']
    else:
        text = sample['rejected']
    split_text = text.split('\n\nAssistant:')
    sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
    sample['completion'] = split_text[-1].strip()


    sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' ' 
    sample['prompt_with_score'] += '<rm1_score>' + ' ' + str(round(sample['helpful_reward'], 1)) + ' '
    sample['prompt_with_score'] += '<rm2_score>' + ' ' + str(round(sample['harmless_reward'], 1)) + ' '
    sample['prompt_with_score'] += '<rm3_score>' + ' ' + str(round(sample['humor_reward'], 1)) + ' '
    
    sample['prompt_with_score'] += '\n\nAssistant:'
    return sample


def template_function_beaver(sample,criteria='better_response_id'):
    assert criteria in ['better_response_id', 'safer_response_id']
    chosen_id = int(sample[criteria])
    sample['completion'] = sample['response_{}'.format(chosen_id)]

    sample['prompt_with_score'] = sample['prompt']+ ' ' 
    sample['prompt_with_score'] += '<rm1_score>' + ' ' + str(round(sample['helpful_reward'], 1)) + ' '
    sample['prompt_with_score'] += '<rm2_score>' + ' ' + str(round(sample['harmless_reward'], 1)) + ' '

    return sample 


def sample_goals(size, num_rewards=3, rewards_list=None, maximum=0.9999):
    if rewards_list is None:
        samples = np.random.normal(0, 1, 100000)
        low, high = np.round(np.quantile(samples, 0), 1), np.round(np.quantile(samples, 1),1)

        preferences = np.round(np.random.random((size, num_rewards)) * (high - low) + low, 1)
        for k in range(len(preferences)):
            min_index = np.argmin(preferences[k])
            for j in range(num_rewards):
                if j != min_index:
                    preferences[k][j] = high
    else: 
        raise NotImplementedError
    return np.round(preferences, 1)


def map_rewards_from_preference(rewards_list, preference, method='linf'):
    n = len(rewards_list)
    target_rewards = np.zeros(n)
    if method == 'linf':
        max_id = np.argmax(preference)
        target_rewards[max_id] = np.round(np.quantile(rewards_list[max_id], 1), 1)
        for i in range(n):
            if i != max_id:
                low, high = np.quantile(rewards_list[i], 0), np.quantile(rewards_list[i], 1)
                target_rewards[i] = np.round(min(n * preference[i], 1) * (high - low) + low, 1)
    elif method == 'l2':
        for i in range(n):
            low, high = np.quantile(rewards_list[i], 0), np.quantile(rewards_list[i], 1)
            target_rewards[i] = np.round(preference[i] / np.sqrt((np.power(preference, 2).sum())) * (high - low) + low, 1)
    elif method == 'linear':
        for i in range(n):
            low, high = np.quantile(rewards_list[i], 0), np.quantile(rewards_list[i], 1)
            target_rewards[i] = np.round(preference[i] * (high - low) + low, 1)
    else:
        raise NotImplementedError
    return target_rewards



# template='''
# Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {instruction}

# ### Response:
# '''
template=dict()
template['empty'] = '''{instruction}'''
template['llama']='''
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST]
'''

template['llama']='''
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{instruction} 
'''



template['vicuna']='''
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {instruction}
'''
template['tulu']='''
<|user|>
{instruction}
<|assistant|>

'''





if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=os.path.join(home,'PLM/gemma-2b'))
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument('--peft_model_path',type=str,default=None)
    parser.add_argument('--max_tokens', type=int, default=128) #max token means max new tokens to generate
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--template', type=str, default='empty')
    parser.add_argument("--preference",type=str,default=None)
    parser.add_argument("--data_file",type=str,default=None, help='the training data for online or test data')
    args = parser.parse_args()


    if args.peft_model_path and not os.path.exists(os.path.join(args.peft_model_path,'peft_intergrated')): 
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,trust_remote_code=True)
        peft_model = PeftModel.from_pretrained(model, args.peft_model_path)
        peft_model = peft_model.merge_and_unload()
        peft_model.save_pretrained(os.path.join(args.peft_model_path,'peft_intergrated'))
        print("PEFT intergrated!!")

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    preference_str = ''
    if args.preference and  ',' in args.preference:
        preference_str= args.preference.replace(',','')
        args.preference = args.preference.split(',')
        args.preference = [float(x) for x in args.preference]
        args.preference = [x/sum(args.preference) for x in args.preference]
        rewards_reference_list = [np.random.randn(50000) for _ in range(len(args.preference))]


    args.data_file = args.data_file.split(',')
        
        

    num_gpus = torch.cuda.device_count()
    #another_args = {'max_num_batched_tokens': args.max_num_batched_tokens} 
    llm = LLM(model =  os.path.join(args.peft_model_path,'peft_intergrated') if args.peft_model_path else args.model_name_or_path,
            tokenizer = args.tokenizer_name_or_path, 
            dtype='bfloat16',
            tensor_parallel_size = num_gpus,
            trust_remote_code=True,
            )
    print('>>>>>> model loaded')

    sampling_params = SamplingParams(temperature = args.temperature, top_p=args.top_p, max_tokens = args.max_tokens)    
    if args.preference is None: # the online generation stage
        raw_dataset = load_dataset('json', data_files = args.data_file)
    else: # the test stage with specified preference 
        raw_dataset = load_dataset('json', data_files = args.data_file)
    # ATTENTION
    raw_dataset = raw_dataset['train']
    if args.preference is None:
        selected_ind = np.random.choice(len(raw_dataset), 20000, replace=False)
        raw_dataset = raw_dataset.select(selected_ind)
    if args.preference:
        target_reward = map_rewards_from_preference(rewards_reference_list, args.preference, method='l2')
        target_reward = np.tile(target_reward, (len(raw_dataset), 1))
    else:
        target_reward = sample_goals(len(raw_dataset), num_rewards=3 if 'HH' in args.data_file[0] else 2)
    
    if 'HH' in args.data_file[0]:
        raw_dataset = raw_dataset.add_column('helpful_reward', target_reward[:,0])
        raw_dataset = raw_dataset.add_column('harmless_reward', target_reward[:,1])
        raw_dataset = raw_dataset.add_column('humor_reward', target_reward[:,2])
    else:
        raw_dataset = raw_dataset.add_column('helpful_reward', target_reward[:,0])
        raw_dataset = raw_dataset.add_column('harmless_reward', target_reward[:,1])
    #raw_dataset = raw_dataset['train'].select(range(0,len(raw_dataset['train']),4))
    templated_dataset = raw_dataset.map(
        template_function_hh if 'HH' in args.data_file[0] else template_function_beaver,
        batched=False,
        remove_columns= ['chosen','rejected'] if 'HH' in args.data_file[0] else None
    )

    # filter out the dataset that is too long or too short
    from transformers import AutoTokenizer
    if 'HH' in args.data_file[0]:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,trust_remote_code=True)
        templated_dataset = templated_dataset.filter(lambda x: len(tokenizer.encode(x['prompt_with_score'])) <= 512 and len(tokenizer.encode(x['prompt_with_score'])) >= 8)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join('/home/futingchen','PLM/gpt2large_harmless_reward'))
        templated_dataset = templated_dataset.filter(lambda x: len(tokenizer.encode(x['prompt_with_score'])) <= 512 and len(tokenizer.encode(x['prompt_with_score'])) >= 8)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join('/home/futingchen','PLM/distilbert_humor_reward'))
        templated_dataset = templated_dataset.filter(lambda x: len(tokenizer.encode(x['prompt_with_score'])) <= 512 and len(tokenizer.encode(x['prompt_with_score'])) >= 8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,trust_remote_code=True)
        templated_dataset = templated_dataset.filter(lambda x: len(tokenizer.encode(x['prompt_with_score'])) <= 512 )
        tokenizer = AutoTokenizer.from_pretrained(os.path.join('/home/tingchen_fu','PLM/gpt2-large'))
        templated_dataset = templated_dataset.filter(lambda x: len(tokenizer.encode(x['prompt_with_score'])) <= 512 )
        
    print(">>>>>> dataset filtered: {}".format(len(templated_dataset)))
    
    #templated_dataset = templated_dataset.select(range(100))

    #template = template['llama'] if 'llama' in args.model_name_or_path.lower() else (template['alpaca'] if 'alpaca' in args.model_name_or_path.lower() else template['tulu'] 

    prompt = [template[args.template].format(instruction = templated_dataset[i]['prompt_with_score']) for i in range(len(templated_dataset))]
    
    print("number of instruction: {}".format(len(prompt)))

    print(prompt[0])
    print(prompt[1])
    print(">>>>>> two cases shown.")
    outputs = llm.generate(prompt, sampling_params)
    sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
    print('>>>>>> generation done')

    if args.peft_model_path:
        args.output_file = os.path.join(os.path.join(root,'dump'),args.peft_model_path.split('/')[-1] + '_'+ args.template)
        args.output_file += '{}.jsonl'.format(preference_str)
    else:
        args.output_file = os.path.join(os.path.join(root,'dump'),args.model_name_or_path.split('/')[-1]+ '_'+ args.template)
        args.output_file += '{}.jsonl'.format(preference_str)

    fout = open(args.output_file,'w')
    for id, output in enumerate(sorted_outputs):
        fout.write(json.dumps({'prompt': templated_dataset[id]['prompt'], 'completion': output.outputs[0].text},ensure_ascii=False)+'\n')
    fout.close()

    # if args.peft_model_path:
    #     os.system('rm -rf '+os.path.join(args.peft_model_path,'peft_intergrated'))
    #     print('>>>>>> PEFT intergrated model removed')