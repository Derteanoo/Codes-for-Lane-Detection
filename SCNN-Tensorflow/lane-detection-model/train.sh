#!/bin/bash

#CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir CULane/list 
#CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir CULane/list --weights_path model/culane_lanenet/culane_scnn
#CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir CULane/list --weights_path model_weights/

CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net mobilev1 --dataset_dir CULane/list 
#CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net mobilev2 --dataset_dir CULane/list 
#CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net shufflev1 --dataset_dir CULane/list 
#CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net shufflev2 --dataset_dir CULane/list 