#!/bin/bash

#CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --weights_path model_weights/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000  --image_path CULane/list/test.txt
CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --net mobilev1 --weights_path model/culane_lanenet/culane_scnn/culane_lanenet_mobilev1_2019-02-12-09-43-12.ckpt-0  --image_path CULane/list/test.txt
