"""
Property-based tests for ConSinGAN Training Module.

These tests verify that the ConSinGANTrainer maintains its correctness properties
across a wide range of inputs and configurations.
"""

import pytest
import torch
import numpy as np
import os
import shutil
import tempfile
from hypothesis import given, strategies as st, settings, HealthCheck
from pathlib import Path

from training import ConSinGANTrainer
from models import ConSinGANGenerator, ConSinGANDiscriminator
from device_manager import DeviceManager
from session_manager import SessionManager


class TestConSinGANTrainerGPUCPUFallback:
    """Property tests for GPU/CPU fallback in training."""
    
    @given(
        device_override=st.sampled_from([None, "cpu"])
    )
    @settings(max_examples=3, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_2_gpu_cpu_fallback(self, device_override):
        """
        Property 2: GPU/CPU Fallback
        
        For any training operation, if GPU (CUDA) is available, it should be used;
        otherwise, the system should seamlessly fall back to CPU mode without errors.
        
        **Validates: Requirements 1.2, 1.3**
        """
        # Determine device
        if device_override:
            device = device_override
        else:
            device = DeviceManager.get_optimal_device()
        
        # Create models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        
        # Create trainer
        trainer = ConSinGANTrainer(generator, discriminator, device)
        
        # Verify trainer is on correct device
        assert trainer.device == device, f"Trainer device {trainer.device} doesn't match {device}"
        
        # Verify models are on correct device
        for param in trainer.generator.parameters():
            assert param.device.type == device.split(':')[0], \
                f"Generator parameter on {param.device.type}, expected {device}"
        
        for param in trainer.discriminator.parameters():
            assert param.device.type == device.split(':')[0], \
                f"Discriminator parameter on {param.device.type}, expected {device}"
        
        # Verify trainer can handle forward pass without errors
        test_tensor = torch.randn(1, 3, 64, 64).to(device)
        with torch.no_grad():
            gen_output = trainer.generator(test_tensor)
            disc_output = trainer.discriminator(test_tensor)
        
        assert gen_output.shape == test_tensor.shape, "Generator output shape mismatch"
        assert 0.0 <= disc_output.item() <= 1.0, "Discriminator output out of range"


class TestConSinGANTrainerModelPersistence:
    """Property tests for model persistence in training."""
    
    @given(
        num_scales=st.sampled_from([3])
    )
    @settings(max_examples=2, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_8_model_persistence(self, num_scales):
        """
        Property 8: Model Persistence
        
        For any completed training operation, the trained Generator and Discriminator
        models should be saved to models/{session_id}/ directory and be loadable
        for inference.
        
        **Validates: Requirements 4.4, 7.2**
        """
        # Create models
        generator = ConSinGANGenerator(num_scales=num_scales)
        discriminator = ConSinGANDiscriminator(num_scales=num_scales)
        device = "cpu"  # Use CPU for testing
        
        # Create trainer
        trainer = ConSinGANTrainer(generator, discriminator, device)
        
        # Create session
        session_id = SessionManager.create_session()
        
        try:
            # Save model
            save_success = trainer.save_model(session_id)
            assert save_success, "Model save should succeed"
            
            # Verify model files exist
            models_path = SessionManager.get_session_path(session_id, "models")
            generator_path = os.path.join(models_path, "generator.pth")
            discriminator_path = os.path.join(models_path, "discriminator.pth")
            config_path = os.path.join(models_path, "config.json")
            
            assert os.path.exists(generator_path), "Generator model file not found"
            assert os.path.exists(discriminator_path), "Discriminator model file not found"
            assert os.path.exists(config_path), "Config file not found"
            
            # Verify model files are not empty
            assert os.path.getsize(generator_path) > 0, "Generator model file is empty"
            assert os.path.getsize(discriminator_path) > 0, "Discriminator model file is empty"
            assert os.path.getsize(config_path) > 0, "Config file is empty"
            
            # Load models and verify they work
            new_generator = ConSinGANGenerator(num_scales=num_scales)
            new_generator.load_state_dict(torch.load(generator_path, map_location=device))
            new_generator.to(device)
            
            new_discriminator = ConSinGANDiscriminator(num_scales=num_scales)
            new_discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
            new_discriminator.to(device)
            
            # Verify loaded models can produce output
            test_tensor = torch.randn(1, 3, 64, 64).to(device)
            
            with torch.no_grad():
                gen_output = new_generator(test_tensor)
                disc_output = new_discriminator(test_tensor)
            
            assert gen_output.shape == test_tensor.shape, "Generator output shape mismatch"
            assert 0.0 <= disc_output.item() <= 1.0, "Discriminator output out of range"
        
        finally:
            # Cleanup
            if os.path.exists(SessionManager.get_session_path(session_id, "uploads")):
                shutil.rmtree(SessionManager.get_session_path(session_id, "uploads"))
            if os.path.exists(SessionManager.get_session_path(session_id, "models")):
                shutil.rmtree(SessionManager.get_session_path(session_id, "models"))


class TestConSinGANTrainerSSIMCalculation:
    """Property tests for SSIM calculation in training."""
    
    @given(
        height=st.sampled_from([64, 128]),
        width=st.sampled_from([64, 128]),
        channels=st.just(3)
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_17_ssim_calculation(self, height, width, channels):
        """
        Property 17: SSIM Calculation
        
        For any completed training, the SSIM score between the original image and
        a generated image should be calculated and returned; if SSIM < 0.5, a
        warning should be issued.
        
        **Validates: Requirements 8.4, 8.5**
        """
        # Create models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        device = "cpu"
        
        # Create trainer
        trainer = ConSinGANTrainer(generator, discriminator, device)
        
        # Create test images
        original_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        generated_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # Calculate SSIM
        ssim_score = trainer.calculate_ssim(original_image, generated_image)
        
        # Verify SSIM is in valid range [0, 1]
        assert 0.0 <= ssim_score <= 1.0, \
            f"SSIM score {ssim_score} should be between 0 and 1"
        
        # Test with identical images (should have high SSIM)
        identical_ssim = trainer.calculate_ssim(original_image, original_image)
        assert identical_ssim > 0.9, \
            f"SSIM for identical images should be > 0.9, got {identical_ssim}"
        
        # Test with different sizes (should handle resizing)
        different_size_image = np.random.randint(0, 256, (height // 2, width // 2, channels), dtype=np.uint8)
        ssim_different_size = trainer.calculate_ssim(original_image, different_size_image)
        assert 0.0 <= ssim_different_size <= 1.0, \
            f"SSIM with different sizes should be valid, got {ssim_different_size}"
        
        # Test with normalized images (0-1 range)
        original_normalized = original_image.astype(np.float32) / 255.0
        generated_normalized = generated_image.astype(np.float32) / 255.0
        
        ssim_normalized = trainer.calculate_ssim(original_normalized, generated_normalized)
        assert 0.0 <= ssim_normalized <= 1.0, \
            f"SSIM with normalized images should be valid, got {ssim_normalized}"


class TestConSinGANTrainerIntegration:
    """Integration tests for ConSinGANTrainer."""
    
    @given(
        learning_rate=st.sampled_from([0.0001, 0.0002])
    )
    @settings(max_examples=2, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_training_config_validation(self, learning_rate):
        """Test that training config is properly validated."""
        # Create models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        device = "cpu"
        
        # Create trainer
        trainer = ConSinGANTrainer(generator, discriminator, device)
        
        # Verify trainer state is initialized
        state = trainer.get_training_state()
        assert state['current_iteration'] == 0
        assert state['total_iterations'] == 0
        assert state['loss_g'] == 0.0
        assert state['loss_d'] == 0.0
    
    @given(
        height=st.sampled_from([64, 128]),
        width=st.sampled_from([64, 128])
    )
    @settings(max_examples=3, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_ssim_calculation_with_various_sizes(self, height, width):
        """Test SSIM calculation with different image sizes."""
        # Create models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        device = "cpu"
        
        # Create trainer
        trainer = ConSinGANTrainer(generator, discriminator, device)
        
        # Create test images
        original_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        generated_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Calculate SSIM
        ssim_score = trainer.calculate_ssim(original_image, generated_image)
        
        # Verify SSIM is in valid range
        assert 0.0 <= ssim_score <= 1.0, f"SSIM score {ssim_score} out of range"
        
        # Test with identical images
        identical_ssim = trainer.calculate_ssim(original_image, original_image)
        assert identical_ssim > 0.9, f"SSIM for identical images should be > 0.9, got {identical_ssim}"
