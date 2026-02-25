#!/bin/bash

# 要执行的Python脚本 
PYTHON_SCRIPT="eval.py"

CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHON_SCRIPT
