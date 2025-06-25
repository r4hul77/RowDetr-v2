#!/bin/bash

PYTHONPATH=${PWD} python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 --master_addr 127.0.0.1 train_scripts/run_dist.py --launcher pytorch
