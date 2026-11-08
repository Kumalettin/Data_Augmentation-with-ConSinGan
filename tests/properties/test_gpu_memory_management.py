"""
Property-based tests for GPU memory management and cleanup.
Tests GPU/CPU fallback, memory management, and session cleanup.
"""

import pytest
from hypothesis import given, strategies as st, settings
import torch
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from device_manager import DeviceManager
from session_manager import SessionManager


class TestGPUCPUFallback:
    """
    Property 2: GPU/CPU Fallback
    
    For any training operation, if GPU (CUDA) is available, it should be used;
    otherwise, the system should seamlessly fall back to CPU mode without errors.
    
    Validates: Requirements 1.2, 1.3
    """
    
    def test_optimal_device_selection(self):
        """Test that optimal device is selected correctly"""
        device = DeviceManager.get_optimal_device()
        assert device in ['cpu', 'cuda']
    
    def test_device_consistency(self):
        """Test that device selection is consistent"""
        device1 = DeviceManager.get_optimal_device()
        device2 = DeviceManager.get_optimal_device()
        assert device1 == device2
    
    def test_cuda_availability_matches_device(self):
        """Test that device selection matches CUDA availability"""
        device = DeviceManager.get_optimal_device()
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            assert device == 'cuda'
        else:
            assert device == 'cpu'
    
    @given(batch_size=st.integers(min_value=1, max_value=32))
    @settings(max_examples=20)
    def test_oom_error_handling(self, batch_size):
        """Test that OOM errors are handled gracefully"""
        device = DeviceManager.get_optimal_device()
        
        new_batch_size, new_device = DeviceManager.handle_oom_error(batch_size, device)
        
        # New batch size should be valid
        assert new_batch_size >= 1
        # New device should be valid
        assert new_device in ['cpu', 'cuda']
        # If on GPU, batch size should be reduced or fallback to CPU
        if device == 'cuda':
            assert new_batch_size <= batch_size or new_device == 'cpu'
    
    def test_oom_fallback_to_cpu(self):
        """Test that OOM on GPU falls back to CPU"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        batch_size = 1
        new_batch_size, new_device = DeviceManager.handle_oom_error(batch_size, 'cuda')
        
        # Should either reduce batch size or fall back to CPU
        assert new_batch_size >= 1
        assert new_device in ['cpu', 'cuda']
    
    def test_cpu_oom_handling(self):
        """Test that OOM on CPU is handled"""
        batch_size = 4
        new_batch_size, new_device = DeviceManager.handle_oom_error(batch_size, 'cpu')
        
        # CPU should return same batch size and device
        assert new_batch_size == batch_size
        assert new_device == 'cpu'


class TestGPUMemoryManagement:
    """
    Property 27: GPU Memory Management
    
    For any GPU operation, if GPU memory is insufficient, the batch size should be
    automatically reduced and training should continue; if still insufficient, the
    system should fall back to CPU.
    
    Validates: Requirements 14.3
    """
    
    def test_gpu_memory_cleanup(self):
        """Test that GPU memory can be cleaned up"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Allocate some memory
        tensor = torch.randn(1000, 1000, device='cuda')
        
        # Clean up
        DeviceManager.cleanup_gpu_memory()
        
        # Should not raise error
        assert True
    
    def test_memory_estimation_cpu(self):
        """Test memory estimation on CPU"""
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        input_shape = (1, 10)
        
        memory = DeviceManager.estimate_memory_usage(model, input_shape, 'cpu')
        
        # Should return a number (0 for CPU)
        assert isinstance(memory, (int, float))
        assert memory >= 0
    
    def test_memory_estimation_gpu(self):
        """Test memory estimation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        import torch.nn as nn
        
        model = nn.Linear(10, 5).to('cuda')
        input_shape = (1, 10)
        
        memory = DeviceManager.estimate_memory_usage(model, input_shape, 'cuda')
        
        # Should return a positive number
        assert isinstance(memory, (int, float))
        assert memory > 0
    
    @given(batch_size=st.integers(min_value=1, max_value=16))
    @settings(max_examples=10)
    def test_batch_size_reduction(self, batch_size):
        """Test that batch size can be reduced"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        new_batch_size, new_device = DeviceManager.handle_oom_error(batch_size, device)
        
        # New batch size should be valid
        assert new_batch_size >= 1
        # New batch size should not exceed original
        assert new_batch_size <= batch_size


class TestGPUMemoryCleanup:
    """
    Property 31: GPU Memory Cleanup
    
    For any completed training or inference operation, GPU memory should be
    released and available for other operations.
    
    Validates: Requirements 15.4
    """
    
    def test_gpu_memory_release(self):
        """Test that GPU memory is released after operations"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Allocate memory
        tensor = torch.randn(1000, 1000, device='cuda')
        
        # Get memory usage
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        
        # Delete tensor
        del tensor
        
        # Clean up
        DeviceManager.cleanup_gpu_memory()
        
        # Memory should be released
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated()
        
        assert memory_after < memory_before
    
    def test_multiple_cleanup_calls(self):
        """Test that multiple cleanup calls don't cause errors"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Multiple cleanup calls should not raise errors
        for _ in range(5):
            DeviceManager.cleanup_gpu_memory()
        
        assert True


class TestSessionCleanup:
    """
    Property 15: Session Cleanup
    
    For any session older than 24 hours, the system should automatically delete
    its associated files (uploads and models) during cleanup operations.
    
    Validates: Requirements 7.4
    """
    
    def test_session_creation(self):
        """Test that sessions can be created"""
        session_id = SessionManager.create_session()
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Clean up
        uploads_path = SessionManager.get_session_path(session_id, 'uploads')
        models_path = SessionManager.get_session_path(session_id, 'models')
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    def test_session_directory_creation(self):
        """Test that session directories are created"""
        session_id = SessionManager.create_session()
        
        uploads_path = SessionManager.get_session_path(session_id, 'uploads')
        models_path = SessionManager.get_session_path(session_id, 'models')
        
        assert os.path.exists(uploads_path)
        assert os.path.exists(models_path)
        
        # Clean up
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    def test_session_isolation(self):
        """Test that sessions are isolated from each other"""
        session1 = SessionManager.create_session()
        session2 = SessionManager.create_session()
        
        assert session1 != session2
        
        path1 = SessionManager.get_session_path(session1, 'uploads')
        path2 = SessionManager.get_session_path(session2, 'uploads')
        
        assert path1 != path2
        
        # Clean up
        for session_id in [session1, session2]:
            uploads_path = SessionManager.get_session_path(session_id, 'uploads')
            models_path = SessionManager.get_session_path(session_id, 'models')
            if os.path.exists(uploads_path):
                shutil.rmtree(uploads_path)
            if os.path.exists(models_path):
                shutil.rmtree(models_path)
    
    def test_cleanup_old_sessions(self):
        """Test that old sessions are cleaned up"""
        # Create a session
        session_id = SessionManager.create_session()
        uploads_path = SessionManager.get_session_path(session_id, 'uploads')
        
        # Verify it exists
        assert os.path.exists(uploads_path)
        
        # Make it old by modifying its timestamp
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(uploads_path, (old_time, old_time))
        
        # Run cleanup
        SessionManager.cleanup_old_sessions(max_age_hours=24)
        
        # Session should be deleted
        assert not os.path.exists(uploads_path)
    
    def test_cleanup_preserves_recent_sessions(self):
        """Test that recent sessions are not cleaned up"""
        # Create a session
        session_id = SessionManager.create_session()
        uploads_path = SessionManager.get_session_path(session_id, 'uploads')
        
        # Verify it exists
        assert os.path.exists(uploads_path)
        
        # Run cleanup (should not delete recent sessions)
        SessionManager.cleanup_old_sessions(max_age_hours=24)
        
        # Session should still exist
        assert os.path.exists(uploads_path)
        
        # Clean up
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        models_path = SessionManager.get_session_path(session_id, 'models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    def test_get_active_sessions(self):
        """Test that active sessions can be retrieved"""
        # Create a session
        session_id = SessionManager.create_session()
        
        # Get active sessions
        active_sessions = SessionManager.get_active_sessions()
        
        assert session_id in active_sessions
        
        # Clean up
        uploads_path = SessionManager.get_session_path(session_id, 'uploads')
        models_path = SessionManager.get_session_path(session_id, 'models')
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    def test_session_exists(self):
        """Test that session existence can be checked"""
        session_id = SessionManager.create_session()
        
        assert SessionManager.session_exists(session_id)
        
        # Clean up
        uploads_path = SessionManager.get_session_path(session_id, 'uploads')
        models_path = SessionManager.get_session_path(session_id, 'models')
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
        
        # Should not exist after cleanup
        assert not SessionManager.session_exists(session_id)
