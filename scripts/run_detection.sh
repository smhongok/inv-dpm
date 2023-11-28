#!/bin/bash

# Ours algorithm
python run_watermark_detection.py --length 10 --num_inference_steps 50 --inv_order 1

# Naive inversion with exact decoder inversion
python run_watermark_detection.py --length 10 --num_inference_steps 50 --inv_order 0

# Naive inversion
python run_watermark_detection.py --length 10 --num_inference_steps 50 --inv_order 0 --wo_decoder_inv