import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, Dict, Tuple
from copy import deepcopy
from loguru import logger
from .trainer import Trainer

class ParallelTrainer:
    """
    Parallel trainer for federated learning clients on single GPU.
    Batches multiple clients together to improve GPU utilization.
    """
    
    def __init__(self, clients: List[Trainer], batch_size: int = 4):
        """
        Initialize parallel trainer.
        
        Args:
            clients: List of client trainers
            batch_size: Number of clients to train in parallel (adjust based on GPU memory)
        """
        self.clients = clients
        self.batch_size = min(batch_size, len(clients))
        self.device = clients[0].device if clients else 'cuda:0'
        
        # Enable memory optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster training
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info(f"Initialized ParallelTrainer with batch_size={self.batch_size} for {len(clients)} clients")
        
    def parallel_local_train(self, n_epoch: int) -> None:
        """
        Train all clients in parallel batches.
        
        Args:
            n_epoch: Number of epochs to train each client
        """
        n_clients = len(self.clients)
        
        # Log initial GPU memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Process clients in batches
        for batch_start in range(0, n_clients, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_clients)
            client_batch = self.clients[batch_start:batch_end]
            
            logger.debug(f"Training clients {batch_start}-{batch_end-1} in parallel")
            
            # Clear cache before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._train_client_batch(client_batch, n_epoch)
            
            # Log memory usage after batch
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                logger.debug(f"Batch {batch_start//self.batch_size}: Current GPU memory: {current_memory:.2f} GB, Peak: {max_memory:.2f} GB")
    
    def _train_client_batch(self, client_batch: List[Trainer], n_epoch: int) -> None:
        """
        Train a batch of clients simultaneously using optimized operations.
        
        Args:
            client_batch: Batch of clients to train
            n_epoch: Number of epochs
        """
        if len(client_batch) == 1:
            # Single client, use original training
            client_batch[0].local_train(n_epoch)
            return
        
        # Use CUDA streams for overlapping computation
        streams = [torch.cuda.Stream() for _ in client_batch] if torch.cuda.is_available() else [None] * len(client_batch)
        
        # Train for specified epochs
        for epoch in range(n_epoch):
            # Train all clients in the batch simultaneously
            for i, (client, stream) in enumerate(zip(client_batch, streams)):
                if stream is not None:
                    with torch.cuda.stream(stream):
                        self._train_client_epoch_optimized(client)
                else:
                    self._train_client_epoch_optimized(client)
            
            # Synchronize all streams
            if torch.cuda.is_available():
                for stream in streams:
                    if stream is not None:
                        stream.synchronize()
            
            # Update schedulers after all clients finish the epoch
            for client in client_batch:
                client.scheduler.step()
    
    def _train_client_epoch_optimized(self, client: Trainer) -> None:
        """
        Optimized training for a single client epoch.
        """
        client.model.train()
        
        # Use gradient accumulation for larger effective batch size
        accumulation_steps = max(1, 512 // client.train_loader.batch_size)  # Target ~512 effective batch size
        client.optimizer.zero_grad()
        
        accumulated_loss = 0.0
        step_count = 0
        
        for batch_idx, (x, y) in enumerate(client.train_loader):
            x, y = x.to(client.device, non_blocking=True), y.to(client.device, non_blocking=True)
            
            # Use mixed precision for faster training and lower memory usage
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    pred = client.model(x)
                    loss = client.criterion(pred, y)
                    loss = loss / accumulation_steps  # Scale loss for accumulation
            else:
                pred = client.model(x)
                loss = client.criterion(pred, y)
                loss = loss / accumulation_steps
            
            loss.backward()
            accumulated_loss += loss.item()
            step_count += 1
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(client.model.parameters(), max_norm=1.0)
                client.optimizer.step()
                client.optimizer.zero_grad()
        
        # Handle remaining gradients
        if step_count % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(client.model.parameters(), max_norm=1.0)
            client.optimizer.step()
            client.optimizer.zero_grad()


class AsyncParallelTrainer:
    """
    Asynchronous parallel trainer for even better GPU utilization.
    """
    
    def __init__(self, clients: List[Trainer], max_concurrent: int = 4):
        """
        Initialize async parallel trainer.
        
        Args:
            clients: List of client trainers
            max_concurrent: Maximum number of concurrent training tasks
        """
        self.clients = clients
        self.max_concurrent = max_concurrent
        self.device = clients[0].device if clients else 'cuda:0'
        
        logger.info(f"Initialized AsyncParallelTrainer with max_concurrent={max_concurrent} for {len(clients)} clients")
    
    async def async_local_train(self, n_epoch: int) -> None:
        """
        Train clients asynchronously.
        """
        import asyncio
        
        # Create semaphore to limit concurrent training
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def train_client_async(client: Trainer):
            async with semaphore:
                # Run training in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, client.local_train, n_epoch)
        
        # Start all training tasks
        tasks = [train_client_async(client) for client in self.clients]
        await asyncio.gather(*tasks)


def create_parallel_trainer(clients: List[Trainer], mode: str = "batch") -> ParallelTrainer:
    """
    Factory function to create appropriate parallel trainer.
    
    Args:
        clients: List of client trainers
        mode: Training mode ("batch" or "async")
    
    Returns:
        Parallel trainer instance
    """
    if mode == "batch":
        # Determine optimal batch size based on GPU memory and number of clients
        n_clients = len(clients)
        
        # Estimate GPU memory usage per client (rough estimation)
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            # Conservative estimate: use 60% of GPU memory, assume each client needs ~2GB
            max_concurrent = max(1, int(total_memory * 0.6 / 2))
        else:
            max_concurrent = 2  # Conservative for CPU
        
        if n_clients <= 4:
            batch_size = min(n_clients, max_concurrent)
        elif n_clients <= 8:
            batch_size = min(4, max_concurrent)
        else:
            batch_size = min(max(2, n_clients // 4), max_concurrent)  # Process in quarters
        
        logger.info(f"Auto-selected batch_size={batch_size} for {n_clients} clients (max_concurrent={max_concurrent})")
        return ParallelTrainer(clients, batch_size)
    elif mode == "async":
        return AsyncParallelTrainer(clients)
    else:
        raise ValueError(f"Unknown mode: {mode}")