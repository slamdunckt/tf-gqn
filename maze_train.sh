#!/bin/sh

python3 train_gqn_draw.py \
  --data_dir /home/cylee/gqn/data/malmo-dataset \
  --dataset minecraft \
  --model_dir /home/cylee/gqn/models/tf-gqn/minecraft/ \
  --seq_length 8 \
  --chkpt_steps 1000 \
  --log_steps 100 \
  --train_epochs 10 \
  --batch_size 36
