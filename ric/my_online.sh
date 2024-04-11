HOME=/home/futingchen
export TRANSFORMERS_CACHE=${HOME}/huggingface_cache
export HF_DATASETS_CACHE=${HOME}/huggingface_cache
export HF_METRICS_CACHE=${HOME}/huggingface_cache


DATE=`date +%m%d%H%M`
RUN_DIR="$PWD"
N_GPU=2

micro_train_bs=2
micro_eval_bs=2
gradient_steps=1
max_grad_norm=0.3
weight_decay=0.01
bs=$(expr $N_GPU \* $gradient_steps \* $micro_train_bs)
warmup_steps=0
warmup_ratio=0
num_train_epochs=1
lr=1e-5
lr_scheduler_type="constant"
eval_strategy="no" #"epoch"
logging_steps=10
save_strategy="epoch"
save_steps=4000 #5000
eval_steps=4000
backbone=Llama-2-7b-hf
peft_type="lora"
lora_modules="gate_proj,up_proj,down_proj"
lora_alpha=16
lora_r=16
lora_dropout=0.05
block_size=512
report_to="none"
metric="loss"
dataset_name='online1_train'


# Prefix to add
data_dir_prefix="${RUN_DIR}/data/HH/"
data_dir_suffix=".jsonl"
# Split the input string by commas
IFS=',' read -ra segments <<< "$dataset_name"
# Add the prefix to each segment and store them in an array
train_file=()
for segment in "${segments[@]}"; do
  train_file+=("${data_dir_prefix}${segment}${data_dir_suffix}")
done
# Join the prefixed segments with commas
train_file=$(IFS=','; echo "${train_file[*]}")
# Print the output string



rm ${HOME}/huggingface_cache/downloads/*.lock
rm ${HOME}/huggingface_cache/*.lock

exp_name=online1_${backbone}_HH_${peft_type}_bf16  # ADD quantitization type at here
exp_setting=bs${bs}lr${lr}decay${weight_decay}${lr_scheduler_type}
SAVE=${RUN_DIR}/dump/${exp_name}_${exp_setting}_${DATE}  #initialization_30
mkdir -p $SAVE

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1234  /home/futingchen/MultiContrast/RiC/ric/my_online.py  \
    --model_name_or_path /home/futingchen/MultiContrast/RiC/ric/dump/offline_Llama-2-7b-hf_HH_lora_bf16_bs4lr1.41e-4decay0.01linear_04102114/peft_intergrated \
    --tokenizer_name_or_path /home/futingchen/PLM/Llama-2-7b-hf \
    --dataset_name  ${dataset_name}  \
    --label_names labels  \
    --train_file  ${train_file}   \
    --data_ratio   1.0  \
    --do_train  \
    --fp16  False    \
    --bf16  True    \
    --load_in_4bit False  \
    --bnb_4bit_quant_type nf4   \
    --bnb_4bit_compute_type  float16  \
    --peft_type ${peft_type} \
    --lora_modules ${lora_modules} \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --lora_dropout ${lora_dropout} \
    --block_size ${block_size}  \
    --per_device_train_batch_size ${micro_train_bs} \
    --per_device_eval_batch_size ${micro_eval_bs} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps 4000 \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_steps} \
    --warmup_ratio ${warmup_ratio} \
    --logging_steps ${logging_steps} \
    --save_total_limit 1 \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${eval_steps} \
    --report_to ${report_to} \
    --run_name ${DATE} \
    --metric_for_best_model ${metric} \
    --output_dir ${SAVE} \
    --ddp_find_unused_parameters False  \
    --overwrite_output_dir  \
    2>&1 | tee ${SAVE}/${DATE}_log.txt

      #--deepspeed ${RUN_DIR}/code/deepspeed_zero3.json  \
      #--optim paged_adamw_32bit  \
      #--cluster_file ${HOME}/ExpertFusion/data/${dataset_name}_cluster4.output  \
      #--cluster_id   ${cluster_id}  \
      #--disable_tqdm "True" \
      #--load_best_model_at_end \
      #--save_steps ${save_steps} \