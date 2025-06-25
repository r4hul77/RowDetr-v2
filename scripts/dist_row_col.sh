#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 --master_addr 127.0.0.1 run_row_col_former.py --launcher pytorch
