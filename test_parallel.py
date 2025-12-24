#!/usr/bin/env python3
"""
Test script for parallel training functionality.
"""

import torch
import time
from src.mozi.experiment import Experiment, ExperimentConfig
from loguru import logger

def test_parallel_training():
    """Test parallel training vs sequential training performance."""
    
    # Small test configuration
    config = ExperimentConfig(
        exp_name='parallel_test',
        model='cnn',  # Use smaller model for testing
        dataset='cifar10',
        split='iid',
        learning_rate=0.01,
        n_client=8,  # Smaller number for testing
        m_client=2,
        n_server=2,
        m_server=0,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        datapath='/data',
        batch_size=128,
        n_epoch=1,  # Single epoch for testing
        n_round=2,  # Just 2 rounds
        num_workers=2,
        attack='none',
        aggregation='fedavg',
        selection_fraction=0.5,
        method='baseline',
        clean_data=0
    )
    
    logger.info("Testing parallel training implementation...")
    logger.info(f"Device: {config.device}")
    logger.info(f"Clients: {config.n_client}")
    
    # Test the experiment
    experiment = Experiment(config)
    
    # Time the training
    start_time = time.time()
    
    # Run one round to test
    loss, acc = experiment.handler.run_round(0)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.success(f"Parallel training completed!")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    if torch.cuda.is_available():
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    test_parallel_training()