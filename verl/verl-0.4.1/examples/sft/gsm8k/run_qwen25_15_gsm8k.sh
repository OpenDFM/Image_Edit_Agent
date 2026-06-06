#!/usr/bin/env bash

CUDA_VISIBLE_DEVICE=0 torchrun -m verl.trainer.fsdp_sft_trainer \
  data.train_files=../../data/gsm8k/processed/train.parquet \
  data.val_files=../../data/gsm8k/processed/test.parquet \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys=['question'] \
  +data.response_dict_keys=['answer'] \
  data.micro_batch_size_per_gpu=1 \
  model.partial_pretrain="../../../models/Qwen2.5-1.5B-Instruct" \
  trainer.project_name=gsm8k-sft \
  trainer.experiment_name=gsm8k-sft-qwen2.5-1.5b \
  trainer.total_epochs=2 \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=1
