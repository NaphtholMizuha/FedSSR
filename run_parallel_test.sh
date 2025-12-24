#!/bin/bash

echo "Testing parallel training implementation..."

# Set CUDA visible devices if needed
export CUDA_VISIBLE_DEVICES=0

# Run the test
python test_parallel.py

echo "Test completed!"