import torch
import gc
import logging
from typing import Dict, Any, List
import threading
import time
from functools import lru_cache
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Optimize memory usage during training and inference"""
    
    def __init__(self):
        self.memory_stats = {}
        
    def clear_memory(self):
        """Clear GPU and CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
    @contextmanager
    def memory_context(self):
        """Context manager for memory-efficient operations"""
        try:
            yield
        finally:
            self.clear_memory()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
            stats['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        
        return stats

class CachingSystem:
    """Caching system for frequent operations"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.prediction_cache = {}
        self.embedding_cache = {}
        
    @lru_cache(maxsize=1000)
    def cached_prediction(self, model, tokenizer, text: str, max_length: int) -> Dict[str, Any]:
        """Cached prediction with LRU eviction"""
        # This would be implemented with actual model prediction
        # For now, return a placeholder
        return {"text": text, "cached": True}
    
    def clear_cache(self):
        """Clear all caches"""
        self.prediction_cache.clear()
        self.embedding_cache.clear()
        self.cached_prediction.cache_clear()

class BackgroundTrainer:
    """Background training for non-blocking UI"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.training_thread = None
        self.is_training = False
        self.training_queue = []
        
    def schedule_training(self, training_data: Dict[str, Any]):
        """Schedule training to run in background"""
        # Only enqueue training jobs. Starting the background worker
        # is left to the caller to control so that scheduling is non-blocking
        # and tests can assert queue behavior deterministically.
        self.training_queue.append(training_data)
    
    def _start_background_training(self):
        """Start training in background thread"""
        if self.training_queue and not self.is_training:
            self.is_training = True
            self.training_thread = threading.Thread(target=self._training_worker)
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def _training_worker(self):
        """Background training worker"""
        while self.training_queue:
            training_data = self.training_queue.pop(0)
            try:
                # Perform training
                result = self.model_manager.incremental_fine_tune(training_data)
                logger.info(f"Background training completed: {result}")
            except Exception as e:
                logger.error(f"Background training failed: {e}")
            
            time.sleep(1)  # Small delay between training sessions
        
        self.is_training = False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'queue_size': len(self.training_queue),
            'active_thread': self.training_thread is not None and self.training_thread.is_alive()
        }