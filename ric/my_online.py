#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset,concatenate_datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# xxx: 2023-04-11
import copy
import json
from transformers import TrainingArguments
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# xxx: 2023-03-21
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default='bfloat16',
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    cluster_dir: Optional[str] = field(default=None, metadata={"help": "The cluster id of each input sentences"}) 
    cluster_id: Optional[str] = field(default=None, metadata={"help": "The cluster id to train on"}) 
    data_ratio: Optional[float] = field(default=1.0, metadata={"help": "The ratio of data to train on"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if ',' in self.dataset_name:
            self.dataset_name = self.dataset_name.split(',')
        else:
            self.dataset_name = [self.dataset_name]
        if ',' in self.train_file:
            self.train_file = self.train_file.split(',')
        else:
            self.train_file = [self.train_file]
        
        if  self.cluster_id and '+' not in self.cluster_id:
            self.cluster_id = [int(self.cluster_id)]
        elif self.cluster_id:
            self.cluster_id = [int(x) for x in self.cluster_id.split('+')]
        
        
        #else:
            # if self.train_file is not None:
            #     extension = self.train_file.split(".")[-1]
            #     assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            # if self.validation_file is not None:
            #     extension = self.validation_file.split(".")[-1]
            #     assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


# # xxx: 2023-04-11, customized args
# @add_start_docstrings(TrainingArguments.__doc__)
@dataclass
class PEFTArguments:
    peft_type: str = field(
        default=None,
        metadata={"help": "Whether to use LoRA."}
    )
    use_int8_training: bool = field(
        default=False,
        metadata={"help": "Whether to use int8 training."}
    )
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_modules: str = field(
        default='q_proj,v_proj',
        metadata={"help": "the distinguishable module names to add lora; splited by commas"}
    )
    lora_dropout: float = field(default = 0.1)


@dataclass
class QuantArguments:
    load_in_4bit: bool = field(
        default=False,
        metadata={"help":'whether to use load in 4bit'}
    )
    bnb_4bit_quant_type: str = field(
        default='nf4',
        metadata={"help": "choose between fp4 and nf4, use nf4 for lora"}
    )
    bnb_4bit_compute_type: str = field(
        default='float32',
        metadata= {"help":"the computation type for bnb  4bit "}
    )
    nested_quant: bool = field(
        default=False,
        metadata={"help": "whether to use double nested quant"}
    )

    
### the peft package is updated
# xxx: save peft adapters at steps/epoch end
# class SavePeftModelCallback(TrainerCallback):
#     def on_save(
#             self,
#             args: TrainingArguments,
#             state: TrainerState,
#             control: TrainerControl,
#             **kwargs,
#     ):
#         checkpoint_folder = os.path.join(
#             args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
#         )

#         peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
#         kwargs["model"].save_pretrained(peft_model_path)

#         # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
#         # if os.path.exists(pytorch_model_path):
#         #     os.remove(pytorch_model_path)
#         return control


# # xxx: save peft at train end
# class SavePeftModelAtEndCallback(TrainerCallback):
#     def on_train_end(
#             self,
#             args: TrainingArguments,
#             state: TrainerState,
#             control: TrainerControl,
#             **kwargs,
#     ):
#         peft_model_path = os.path.join(args.output_dir, "adapter_model")
#         kwargs["model"].save_pretrained(peft_model_path)

#         # pytorch_model_path = os.path.join(state.best_model_checkpoint, "pytorch_model.bin")
#         # if os.path.exists(pytorch_model_path):
#         #     os.remove(pytorch_model_path)
#         return control


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # xxx: 2023-04-12
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PEFTArguments, QuantArguments,  TrainingArguments ))
    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, peft_args, quant_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, peft_args, quant_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"data parameters {data_args}")
    
    # torch.save(data_args.__dict__, os.path.join(training_args.output_dir,'data_args.json'),'w')
    # torch.save(quant_args.__dict__, os.path.join(training_args.output_dir,'quant_args.json'),'w')
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # prepare the dataset
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        assert len(data_args.train_file) == len(data_args.dataset_name)
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    if data_args.cluster_dir is not None:
        from datasets import DatasetDict
        count_one_cluster=[]
        count_all_cluster=[]
        one_cluster_all_dataset = []
        for dataset_name,train_file in zip(data_args.dataset_name,data_args.train_file):
            single_dataset_all_cluster = load_dataset(
                'json',
                data_files=train_file,
            )
            cluster_file = os.path.join(data_args.cluster_dir,dataset_name+'.json')
            assert os.path.exists(cluster_file)
            cluster = json.load(open(cluster_file))
            assert len(cluster) == len(single_dataset_all_cluster['train'])
            keep_indices = [x for x in range(len(single_dataset_all_cluster['train'])) if cluster[x] in data_args.cluster_id ] 
            single_dataset_one_cluster = single_dataset_all_cluster['train'].select(keep_indices)
            one_cluster_all_dataset.append(single_dataset_one_cluster)
            count_one_cluster.append(len(single_dataset_one_cluster))
            count_all_cluster.append(len(single_dataset_all_cluster['train']))
        assert len(one_cluster_all_dataset) == len(data_args.dataset_name)
        raw_datasets = DatasetDict({'train':concatenate_datasets(one_cluster_all_dataset)})
        assert len(raw_datasets['train']) == sum(count_one_cluster)
    else:
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            #load_from_cache_file=False
        )

    raw_datasets = raw_datasets.shuffle(seed=training_args.seed)
    if data_args.data_ratio != 1.0:
        raw_datasets['train'] = raw_datasets['train'].select(range(int(len(raw_datasets['train'])*data_args.data_ratio)))

    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys() and training_args.do_eval:
        logger.info("NOTICE: split a validation set from the training set.")
        raw_datasets["validation"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            #cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            #cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
    # if data_args.cluster_file and data_args.cluster_id is not None:
    #     assert os.path.exists(data_args.cluster_file)
    #     cluster = json.load(open(data_args.cluster_file))
    #     keep_indices = [x for x in range(len(raw_datasets['train'])) if cluster[x] == int(data_args.cluster_id)]
    #     raw_datasets['train'] = raw_datasets['train'].select(keep_indices)
    #     print("cluster {} count {}".format(data_args.cluster_id, len(raw_datasets['train'])))


    ## prepare the config
    config_kwargs = {
       # "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,trust_remote_code=True, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,trust_remote_code=True, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")


    if 'baichuan' in model_args.model_name_or_path:
        config.z_loss_weight=0.0

    ### prepare the tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,   **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # xxx: 2023-03-21, add padding


    if tokenizer.pad_token is None:
        print("tokenizer pad_token is None, use eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    if 'mistral' in model_args.model_name_or_path:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = 'right'



    ## prepare the model 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_args.load_in_4bit,
        bnb_4bit_quant_type = quant_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype =  quant_args.bnb_4bit_compute_type,
        bnb_4bit_use_double_quant= quant_args.nested_quant,
    )
    logger.info(f"bng configs: {bnb_config}")
    if 'phi' in model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        from phi_model.modeling_phi import PhiForCausalLM
        model = PhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.model_name_or_path,
            use_auth_token=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            quantization_config=bnb_config if quant_args.load_in_4bit else None,
            
        )
    elif model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # xxx: 2023-04-11, LoRA
        # xxx: int8 is not compatible with DeepSpeed (require not to pass device_map)
        # xxx: 8bit models should not be converted to DDP
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/mpt-7b',
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config if quant_args.load_in_4bit else None,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # xxx: 2023-04-11, setup LoRA
    if peft_args.peft_type == "lora":
        lora_config = LoraConfig(
            r = peft_args.lora_r,
            lora_alpha = peft_args.lora_alpha,
            target_modules=peft_args.lora_modules.split(','),
            lora_dropout=peft_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA configs: {lora_config}")
        # xxx: To avoid "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        # xxx: Seems due to gradient_checkpointing, to check later
        #TODO: what is this for??
        # if hasattr(model, "enable_input_require_grads"):
        #     model.enable_input_require_grads()
        # else:
        #     def make_inputs_require_grad(module, input, output):
        #         output.requires_grad_(True)

        #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        print(model.dtype)
        setattr(model, 'model_parallel', True) # 
        setattr(model, 'is_parallelizable', True)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = list(raw_datasets["train"].features)
    # else:
    #     column_names = list(raw_datasets["validation"].features)
    #text_column_name = "text" if "text" in column_names else column_names[0]

    # if data_args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > 1024:
    #         logger.warning(
    #             "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
    #             " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
    #             " override this default with `--block_size xxx`."
    #         )
    #         block_size = 1024
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)

    # xxx: 2023-03-14
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


    def template_function_hh(sample,chosen=True):
        sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' ' 
        sample['prompt_with_score'] += '<rm1_score>' + ' ' + str(round(sample['helpful_reward'], 1)) + ' '
        sample['prompt_with_score'] += '<rm2_score>' + ' ' + str(round(sample['harmless_reward'], 1)) + ' '
        sample['prompt_with_score'] += '<rm3_score>' + ' ' + str(round(sample['humor_reward'], 1)) + ' '
        
        sample['prompt_with_score'] += '\n\nAssistant:'

        return sample




    def completion_preprocess_function(example,  add_bos=False):
        '''
        Here we assume each example has 'prompt' and 'completion' fields.
        We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
        and it doesn't make sense to follow directly with the completion.
        '''
        #### if prompt doesn't end with space and completion doesn't start with space, add space
        if not example['prompt_with_score'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
            example_text = example['prompt_with_score'] + ' ' + example['completion']
        else:
            example_text = example['prompt_with_score'] + example['completion']
        example_text = example_text + tokenizer.eos_token
        if add_bos:
            example_text = tokenizer.bos_token + example_text
        
        tokenized_example = tokenizer(example_text, return_tensors='pt', )# No max length here and we afterward filter them out.
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
        tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt',)
        # mask the prompt part for avoiding loss
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }

    def message_preprocess_function(example, tokenizer, max_seq_length, add_bos=False):
        '''
        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        '''
        messages = example['messages']
        if len(messages) == 0:
            raise ValueError('messages field is empty.')
        
        def _concat_messages(messages):
            message_text = ""
            for message in messages:
                if message["role"] == "system":
                    message_text += "<|system|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "user":
                    message_text += "<|user|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "assistant":
                    message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
            return message_text
            
        example_text = _concat_messages(messages).strip()
        if add_bos:
            example_text = tokenizer.bos_token + example_text
        tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[:message_idx+1])
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors='pt', 
                    max_length=max_seq_length, 
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100
                
                if message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map



    with training_args.main_process_first(desc="example per line with padding"):
        if not data_args.streaming:
            templated_datasets = raw_datasets.map(
                template_function_hh,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                #load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenize with padding",
            )
        else:
            templated_datasets = raw_datasets.map(
                template_function_hh,
                batched=False,
            )


    with training_args.main_process_first(desc="example per line with padding"):
        if not data_args.streaming:
            lm_datasets = templated_datasets.map(
                completion_preprocess_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                #load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenize with padding",
            )
        else:
            lm_datasets = templated_datasets.map(
                completion_preprocess_function,
                batched=False,
            )

    with training_args.main_process_first(desc="example per line with padding"):
        lm_datasets = lm_datasets.filter(lambda x: len(x['input_ids']) <= data_args.block_size and  len(x['input_ids']) >= 8)

    


    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # xxx: print samples
        logger.info("xxx: Show the number of tokenized training sample {}. ".format(len(train_dataset)))
        if data_args.cluster_dir is not None:
            logger.info("xxx: The number of training samples in cluster {} is {}:{} out of {}:{}.".format(data_args.cluster_id,len(train_dataset),count_one_cluster,sum(count_all_cluster),count_all_cluster))

    if training_args.do_eval:
        #if "validation" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # xxx: 2023-04-13, load pretrained adapter weights
    if training_args.do_train:
        checkpoint = None
        # resume_from_checkpoint is prefered
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # xxx: checkpoint is a folder
        if checkpoint:
            peft_model_name = os.path.join(checkpoint, "adapter_model.bin")
            if os.path.exists(peft_model_name):
                logger.info(f"xxx: Load pretrained adapter weights from {peft_model_name}")
                adapters_weights = torch.load(peft_model_name)
                set_peft_model_state_dict(model, adapters_weights)
                logger.info(f"xxx: Double check the trainable parameters...")
                model.print_trainable_parameters()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        #data_collator=default_data_collator,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                          padding=True, label_pad_token_id=IGNORE_INDEX),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        #callbacks=[SavePeftModelCallback, SavePeftModelAtEndCallback] if training_args.use_lora else None,     # xxx: 2023-04-12, callbacks for
    )
    # xxx: 2023-04-11, LoRA
    if peft_args.peft_type == "lora":
        model.config.use_cache = False

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()


if __name__ == "__main__":
    main()
