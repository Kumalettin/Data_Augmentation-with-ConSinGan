"""
Device management for GPU/CPU fallback and memory optimization.
"""

import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device selection (GPU/CPU) and handles memory-related operations.
    Provides GPU/CPU fallback mechanism and memory estimation.
    """

    @staticmethod
    def get_optimal_device():
        """
        Detect and return the optimal device for computation.
        
        Returns:
            str: "cuda" if GPU is available, "cpu" otherwise
        """
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return "cuda"
        logger.info("GPU not available, using CPU")
        return "cpu"

    @staticmethod
    def handle_oom_error(batch_size, device):
        """
        Handle out-of-memory errors by reducing batch size or falling back to CPU.
        
        Args:
            batch_size (int): Current batch size
            device (str): Current device ("cuda" or "cpu")
        
        Returns:
            tuple: (new_batch_size, new_device)
        """
        if device == "cuda":
            torch.cuda.empty_cache()
            new_batch_size = max(1, batch_size // 2)
            
            if new_batch_size < batch_size:
                logger.warning(f"GPU OOM: reducing batch size from {batch_size} to {new_batch_size}")
                return new_batch_size, "cuda"
            else:
                # Fall back to CPU
                logger.warning("GPU OOM: falling back to CPU")
                return 1, "cpu"
        
        return batch_size, device

    @staticmethod
    def estimate_memory_usage(model, input_shape, device):
        """
        Estimate GPU memory usage for a given model and input shape.
        
        Args:
            model: PyTorch model
            input_shape (tuple): Input tensor shape (e.g., (1, 3, 64, 64))
            device (str): Device to estimate for
        
        Returns:
            float: Estimated memory usage in MB, or float('inf') if estimation fails
        """
        try:
            input_tensor = torch.randn(*input_shape).to(device)
            
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            if device == "cuda":
                torch.cuda.synchronize()
                memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            else:
                memory_used = 0  # CPU memory estimation not implemented
            
            return memory_used
        except RuntimeError as e:
            logger.error(f"Memory estimation failed: {e}")
            return float('inf')

    @staticmethod
    def cleanup_gpu_memory():
        """
        Clean up GPU memory after operations.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cleaned up")
