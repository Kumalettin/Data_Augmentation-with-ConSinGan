"""
Property-based tests for PyTorch library loading and device detection.
Tests that PyTorch and torchvision are properly loaded and device detection works.
"""

import pytest
from hypothesis import given, strategies as st, settings
import sys


class TestPyTorchLibraryLoading:
    """
    Property 1: PyTorch Library Loading
    
    For any backend initialization, PyTorch and torchvision libraries should be
    successfully loaded and available for use.
    
    Validates: Requirements 1.1, 1.2, 1.3
    """
    
    def test_pytorch_import(self):
        """Test that PyTorch can be imported"""
        try:
            import torch
            assert torch is not None
            assert hasattr(torch, '__version__')
        except ImportError:
            pytest.fail("PyTorch import failed")
    
    def test_torchvision_import(self):
        """Test that torchvision can be imported"""
        try:
            import torchvision
            assert torchvision is not None
            assert hasattr(torchvision, '__version__')
        except ImportError:
            pytest.fail("Torchvision import failed")
    
    def test_pytorch_version_valid(self):
        """Test that PyTorch version is valid"""
        import torch
        version = torch.__version__
        assert isinstance(version, str)
        assert len(version) > 0
        # Version should contain at least major.minor
        assert '.' in version
    
    def test_torchvision_version_valid(self):
        """Test that torchvision version is valid"""
        import torchvision
        version = torchvision.__version__
        assert isinstance(version, str)
        assert len(version) > 0
    
    def test_torch_tensor_creation(self):
        """Test that PyTorch can create tensors"""
        import torch
        tensor = torch.randn(2, 3)
        assert tensor.shape == (2, 3)
        assert tensor.dtype == torch.float32
    
    def test_torch_cuda_availability(self):
        """Test that CUDA availability can be checked"""
        import torch
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)
    
    def test_torch_device_creation(self):
        """Test that torch devices can be created"""
        import torch
        
        # CPU device should always work
        cpu_device = torch.device('cpu')
        assert str(cpu_device) == 'cpu'
        
        # CUDA device creation should not raise error
        # (even if CUDA is not available)
        try:
            cuda_device = torch.device('cuda')
            assert 'cuda' in str(cuda_device)
        except RuntimeError:
            # CUDA not available, which is fine
            pass


class TestDeviceDetection:
    """Test device detection functionality"""
    
    def test_device_manager_import(self):
        """Test that DeviceManager can be imported"""
        try:
            from device_manager import DeviceManager
            assert DeviceManager is not None
        except ImportError:
            pytest.fail("DeviceManager import failed")
    
    def test_device_manager_instantiation(self):
        """Test that DeviceManager can be instantiated"""
        from device_manager import DeviceManager
        manager = DeviceManager()
        assert manager is not None
    
    def test_get_optimal_device(self):
        """Test that get_optimal_device returns a valid device"""
        from device_manager import DeviceManager
        manager = DeviceManager()
        device = manager.get_optimal_device()
        
        assert device is not None
        assert isinstance(device, str)
        assert device in ['cpu', 'cuda']
    
    def test_device_consistency(self):
        """Test that device detection is consistent"""
        from device_manager import DeviceManager
        manager = DeviceManager()
        
        device1 = manager.get_optimal_device()
        device2 = manager.get_optimal_device()
        
        assert device1 == device2


class TestGPUCPUFallback:
    """
    Property 2: GPU/CPU Fallback
    
    For any training operation, if GPU (CUDA) is available, it should be used;
    otherwise, the system should seamlessly fall back to CPU mode without errors.
    
    Validates: Requirements 1.2, 1.3
    """
    
    def test_cuda_availability_check(self):
        """Test that CUDA availability can be checked"""
        import torch
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)
    
    def test_device_fallback_logic(self):
        """Test that device fallback logic works"""
        import torch
        
        # Try to use CUDA, fall back to CPU if not available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert device is not None
        
        # Create a tensor on the device
        tensor = torch.randn(2, 3, device=device)
        assert tensor.device.type in ['cpu', 'cuda']
    
    def test_model_to_device(self):
        """Test that models can be moved to device"""
        import torch
        import torch.nn as nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simple model
        model = nn.Linear(10, 5)
        model = model.to(device)
        
        # Check that model is on correct device
        assert next(model.parameters()).device.type == device.type
    
    def test_tensor_operations_on_device(self):
        """Test that tensor operations work on both CPU and GPU"""
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create tensors
        x = torch.randn(10, 5, device=device)
        y = torch.randn(5, 3, device=device)
        
        # Perform operations
        z = torch.matmul(x, y)
        
        assert z.shape == (10, 3)
        assert z.device.type == device.type


class TestBackendInitialization:
    """Test backend initialization with PyTorch"""
    
    def test_backend_imports(self):
        """Test that backend can import all required modules"""
        try:
            import torch
            import torchvision
            from device_manager import DeviceManager
            from models import ConSinGANGenerator, ConSinGANDiscriminator
            from training import ConSinGANTrainer, TrainingTaskManager
            from inference import ConSinGANInference
            from session_manager import SessionManager
            
            assert all([
                torch,
                torchvision,
                DeviceManager,
                ConSinGANGenerator,
                ConSinGANDiscriminator,
                ConSinGANTrainer,
                TrainingTaskManager,
                ConSinGANInference,
                SessionManager
            ])
        except ImportError as e:
            pytest.fail(f"Backend import failed: {e}")
    
    def test_device_manager_initialization(self):
        """Test that DeviceManager initializes correctly"""
        from device_manager import DeviceManager
        
        manager = DeviceManager()
        device = manager.get_optimal_device()
        
        assert device is not None
        assert isinstance(device, str)
        assert device in ['cpu', 'cuda']
    
    def test_generator_creation_on_device(self):
        """Test that Generator can be created on detected device"""
        import torch
        from device_manager import DeviceManager
        from models import ConSinGANGenerator
        
        manager = DeviceManager()
        device_str = manager.get_optimal_device()
        device = torch.device(device_str)
        
        generator = ConSinGANGenerator(num_scales=3)
        generator = generator.to(device)
        
        assert next(generator.parameters()).device.type == device.type
    
    def test_discriminator_creation_on_device(self):
        """Test that Discriminator can be created on detected device"""
        import torch
        from device_manager import DeviceManager
        from models import ConSinGANDiscriminator
        
        manager = DeviceManager()
        device_str = manager.get_optimal_device()
        device = torch.device(device_str)
        
        discriminator = ConSinGANDiscriminator(num_scales=3)
        discriminator = discriminator.to(device)
        
        assert next(discriminator.parameters()).device.type == device.type


class TestLibraryVersions:
    """Test that library versions are compatible"""
    
    def test_pytorch_minimum_version(self):
        """Test that PyTorch version is at least 1.9"""
        import torch
        version_parts = torch.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        # PyTorch 1.9 or higher
        assert major > 1 or (major == 1 and minor >= 9)
    
    def test_python_version_compatibility(self):
        """Test that Python version is compatible"""
        import sys
        # Python 3.8 or higher
        assert sys.version_info >= (3, 8)
    
    def test_numpy_available(self):
        """Test that numpy is available"""
        try:
            import numpy as np
            assert np is not None
        except ImportError:
            pytest.fail("NumPy import failed")
    
    def test_opencv_available(self):
        """Test that OpenCV is available"""
        try:
            import cv2
            assert cv2 is not None
        except ImportError:
            pytest.fail("OpenCV import failed")
    
    def test_pil_available(self):
        """Test that PIL is available"""
        try:
            from PIL import Image
            assert Image is not None
        except ImportError:
            pytest.fail("PIL import failed")
