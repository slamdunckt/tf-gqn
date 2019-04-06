#!/bin/sh

python3 train_gqn_draw.py \
  --data_dir /media/cylee/StorageDisk/gqn-dataset/ \
  --dataset mazes \
  --model_dir /home/cylee/gqn/models/tf-gqn/maze_attention_fix/ \
  --seq_length 8 \
  --chkpt_steps 1000 \
  --log_steps 100 \
  --train_epochs 5000 \
  --batch_size 13
