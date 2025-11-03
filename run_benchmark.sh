#!/bin/bash
export LOG_LEVEL=INFO
python3 /home/tuomo/code/vllm_gateway/performance_analysis.py 2>&1 | tail -300
