#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Sun 26 Jan 2020 08:21:24 PM CST
#########################################################################
#!/bin/bash
WORKERS=0
MODEL='ac_resnet50_normal'
PYTHONPATH=$HOME/asybn/ \
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=8 \
    train.py \
    --world-size=8 \
    --model $MODEL \
    --sync-bn \
    --epochs 120 \
    --data-path=$HOME/datasets/imagenet \
    --output-dir=checkpoints/$MODEL \
    --batch-size=64 \
    --print-freq=100 \
    --workers=$WORKERS \
    |& tee -a checkpoints/$MODEL/train-$MODEL.log
