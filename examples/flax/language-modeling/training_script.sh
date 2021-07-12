export model_dir=arabic-t5-base
export train_batch_size=2048
export eval_batch_size=4096

python ./run_t5_mlm_flax.py \
--model_type t5 \
--config_name ${model_dir} \
--tokenizer_name ${model_dir} \
--use_fast_tokenizer True \
--dtype bfloat16 \
--max_seq_length 512 \
--preprocessing_num_workers 96 \
--output_dir ${model_dir} \
--overwrite_output_dir True \
--do_train \
--eval_steps 5000 \
--per_device_train_batch_size ${train_batch_size} \
--per_device_eval_batch_size ${eval_batch_size} \
--learning_rate 1e-2 \
--num_train_epochs 1 \
--logging_steps 500 \
--save_steps 10000 \
--seed 12 \
--adafactor True \
--push_to_hub \
--cache_dir ./training_cache \
--save_total_limit 5
