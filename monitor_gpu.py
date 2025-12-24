#!/usr/bin/env python3
"""
GPU monitoring script to track utilization during parallel training.
"""

import torch
import time
import threading
from loguru import logger
import psutil
import os

class GPUMonitor:
    """Monitor GPU utilization and memory usage."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.stats = []
        
    def start_monitoring(self):
        """Start monitoring in a separate thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("GPU monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        logger.info("GPU monitoring stopped")
        return self.get_stats()
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            if torch.cuda.is_available():
                # Get GPU stats
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                
                # Try to get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    memory_util = utilization.memory
                except ImportError:
                    gpu_util = -1  # Not available
                    memory_util = -1
                
                # Get CPU stats
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                stat = {
                    'timestamp': time.time(),
                    'gpu_memory_allocated': memory_allocated,
                    'gpu_memory_reserved': memory_reserved,
                    'gpu_utilization': gpu_util,
                    'gpu_memory_utilization': memory_util,
                    'cpu_percent': cpu_percent,
                    'system_memory_percent': memory_percent
                }
                
                self.stats.append(stat)
                
                # Log current stats
                if gpu_util >= 0:
                    logger.debug(f"GPU Util: {gpu_util}%, GPU Mem: {memory_allocated:.2f}GB, CPU: {cpu_percent:.1f}%")
                else:
                    logger.debug(f"GPU Mem: {memory_allocated:.2f}GB, CPU: {cpu_percent:.1f}%")
            
            time.sleep(self.interval)
    
    def get_stats(self):
        """Get monitoring statistics."""
        if not self.stats:
            return {}
            
        gpu_utils = [s['gpu_utilization'] for s in self.stats if s['gpu_utilization'] >= 0]
        gpu_mems = [s['gpu_memory_allocated'] for s in self.stats]
        cpu_utils = [s['cpu_percent'] for s in self.stats]
        
        stats = {
            'duration': self.stats[-1]['timestamp'] - self.stats[0]['timestamp'],
            'samples': len(self.stats),
            'avg_gpu_memory': sum(gpu_mems) / len(gpu_mems) if gpu_mems else 0,
            'max_gpu_memory': max(gpu_mems) if gpu_mems else 0,
            'avg_cpu_util': sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0,
        }
        
        if gpu_utils:
            stats.update({
                'avg_gpu_util': sum(gpu_utils) / len(gpu_utils),
                'max_gpu_util': max(gpu_utils),
                'min_gpu_util': min(gpu_utils)
            })
        
        return stats

def run_monitored_training():
    """Run training with GPU monitoring."""
    from src.mozi.experiment import Experiment, ExperimentConfig
    
    # Configuration for monitoring test
    config = ExperimentConfig(
        exp_name='gpu_monitor_test',
        model='resnet',
        dataset='cifar100',
        split='iid',
        learning_rate=0.1,
        n_client=20,
        m_client=8,
        n_server=5,
        m_server=0,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        datapath='/data',
        batch_size=256,  # Larger batch size for better GPU utilization
        n_epoch=2,  # Reduced for testing
        n_round=3,  # Just a few rounds
        num_workers=4,
        attack='min-max-std',
        aggregation='fedavg',
        selection_fraction=0.5,
        method='baseline',  # Use parallel training
        clean_data=100
    )
    
    logger.info("Starting monitored training...")
    logger.info(f"Configuration: {config.n_client} clients, batch_size={config.batch_size}")
    
    # Initialize monitoring
    monitor = GPUMonitor(interval=0.5)  # Monitor every 0.5 seconds
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Create and run experiment
        experiment = Experiment(config)
        
        # Run a few rounds
        for round_num in range(config.n_round):
            logger.info(f"Starting round {round_num}")
            loss, acc = experiment.handler.run_round(round_num)
            logger.info(f"Round {round_num}: Loss={loss:.4f}, Acc={acc:.4f}")
            
    finally:
        # Stop monitoring and get stats
        stats = monitor.stop_monitoring()
        
        # Print summary
        logger.success("Training completed! GPU utilization summary:")
        logger.info(f"Duration: {stats.get('duration', 0):.1f} seconds")
        logger.info(f"Samples: {stats.get('samples', 0)}")
        logger.info(f"Average GPU Memory: {stats.get('avg_gpu_memory', 0):.2f} GB")
        logger.info(f"Peak GPU Memory: {stats.get('max_gpu_memory', 0):.2f} GB")
        logger.info(f"Average CPU Utilization: {stats.get('avg_cpu_util', 0):.1f}%")
        
        if 'avg_gpu_util' in stats:
            logger.info(f"Average GPU Utilization: {stats['avg_gpu_util']:.1f}%")
            logger.info(f"Peak GPU Utilization: {stats['max_gpu_util']:.1f}%")
            logger.info(f"Min GPU Utilization: {stats['min_gpu_util']:.1f}%")
        else:
            logger.warning("GPU utilization monitoring not available (install pynvml: pip install nvidia-ml-py)")

if __name__ == "__main__":
    run_monitored_training()