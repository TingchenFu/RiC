HOME=/home/futingchen
# CUDA_VISIBLE_DEVICES=0,1  python -u /home/futingchen/MultiContrast/code/vllm_generate.py  \
#   --model_name_or_path  ${HOME}/PLM/Llama-2-7b-hf  \
#   --template empty \

# CUDA_VISIBLE_DEVICES=0,1  python -u /home/futingchen/MultiContrast/code/vllm_generate.py  \
#   --model_name_or_path  ${HOME}/PLM/Llama-2-7b-hf  \
#   --template vicuna \


# CUDA_VISIBLE_DEVICES=0,1  python -u /home/futingchen/MultiContrast/code/vllm_generate.py  \
#   --model_name_or_path  ${HOME}/PLM/Llama-2-7b-hf  \
#   --peft_model_path  /home/futingchen/MultiContrast/dump/SFT_Llama-2-7b-hf_HH_lora_bf16_bs8lr3e-4decay0.0cosine  \
#   --template vicuna \

# CUDA_VISIBLE_DEVICES=0,1  python -u /home/futingchen/MultiContrast/code/vllm_generate.py  \
#   --model_name_or_path  ${HOME}/PLM/Llama-2-7b-hf  \
#   --peft_model_path  /home/futingchen/MultiContrast/dump/SFT_Llama-2-7b-hf_HH_lora_bf16_bs8lr3e-4decay0.0cosine  \
#   --template empty \

# CUDA_VISIBLE_DEVICES=0,1  python -u /home/futingchen/MultiContrast/code/vllm_generate.py  \
#   --model_name_or_path  ${HOME}/PLM/Llama-2-7b-hf  \
#   --peft_model_path  /home/futingchen/MultiContrast/dump/SFT_Llama-2-7b-hf_HH_lora_bf16_bs16lr3e-4decay0.0cosine  \
#   --template empty \



CUDA_VISIBLE_DEVICES=2,3  python -u /home/futingchen/MultiContrast/RiC/ric/my_generation.py  \
  --tokenizer_name_or_path   ${HOME}/PLM/Llama-2-7b-hf  \
  --model_name_or_path  /home/futingchen/MultiContrast/RiC/ric/dump/offline_Llama-2-7b-hf_HH_lora_bf16_bs4lr1.41e-4decay0.01linear_04102114/peft_intergrated  \
  --peft_model_path  /home/futingchen/MultiContrast/RiC/ric/dump/online1_Llama-2-7b-hf_HH_lora_bf16_bs4lr1e-5decay0.01constant_04111305  \
  --template empty \