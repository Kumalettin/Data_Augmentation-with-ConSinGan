"""
Property-based tests for ConSinGAN inference module.

These tests verify that the inference module correctly loads models,
generates images, and encodes them as base64.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, HealthCheck
from inference import ConSinGANInference
from models import ConSinGANGenerator, ConSinGANDiscriminator
from session_manager import SessionManager
from device_manager import DeviceManager


class TestConSinGANInferenceModelLoading:
    """Property tests for model loading in ConSinGANInference."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Create a temporary session for testing
        self.test_session_id = SessionManager.create_session()
        
        # Create and save dummy models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        torch.save(generator, os.path.join(models_path, "generator.pth"))
        torch.save(discriminator, os.path.join(models_path, "discriminator.pth"))
        
        yield
        
        # Cleanup
        uploads_path = SessionManager.get_session_path(self.test_session_id, "uploads")
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    @given(
        device=st.sampled_from(["cpu"])  # Use CPU for testing
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_11_inference_model_loading(self, device):
        """
        Property 11: Inference Model Loading
        
        For any inference request, if a trained model exists for the session,
        it should be loaded successfully; otherwise, an appropriate error
        should be returned.
        
        **Validates: Requirements 6.1, 6.5**
        """
        # Test 1: Load existing model successfully
        inference = ConSinGANInference(self.test_session_id, device=device)
        
        # Should load without error
        result = inference.load_model(self.test_session_id)
        assert result is True, "Model loading should return True"
        assert inference.generator is not None, "Generator should be loaded"
        assert inference.discriminator is not None, "Discriminator should be loaded"
        
        # Test 2: Try to load non-existent model
        non_existent_session = "non-existent-session-id"
        inference2 = ConSinGANInference(non_existent_session, device=device)
        
        with pytest.raises(FileNotFoundError):
            inference2.load_model(non_existent_session)
    
    @given(
        device=st.sampled_from(["cpu"])
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_11_inference_model_loading_device_placement(self, device):
        """
        Property 11: Inference Model Loading (Device Placement)
        
        Verify that loaded models are correctly placed on the specified device.
        
        **Validates: Requirements 6.1, 6.5**
        """
        inference = ConSinGANInference(self.test_session_id, device=device)
        inference.load_model(self.test_session_id)
        
        # Verify models are on correct device
        for param in inference.generator.parameters():
            assert param.device.type == device, f"Generator should be on {device}"
            break
        
        for param in inference.discriminator.parameters():
            assert param.device.type == device, f"Discriminator should be on {device}"
            break


class TestConSinGANInferenceSyntheticImageGeneration:
    """Property tests for synthetic image generation."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Create a temporary session for testing
        self.test_session_id = SessionManager.create_session()
        
        # Create and save dummy models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        torch.save(generator, os.path.join(models_path, "generator.pth"))
        torch.save(discriminator, os.path.join(models_path, "discriminator.pth"))
        
        yield
        
        # Cleanup
        uploads_path = SessionManager.get_session_path(self.test_session_id, "uploads")
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    @given(
        num_images=st.integers(min_value=1, max_value=10),
        device=st.sampled_from(["cpu"])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_12_synthetic_image_generation_count(self, num_images, device):
        """
        Property 12: Synthetic Image Generation Count
        
        For any inference request with num_images parameter, the system should
        generate exactly that many synthetic images (or return an error if impossible).
        
        **Validates: Requirements 6.2**
        """
        inference = ConSinGANInference(self.test_session_id, device=device)
        inference.load_model(self.test_session_id)
        
        # Generate images
        images = inference.generate(num_images=num_images)
        
        # Verify correct number of images generated
        assert len(images) == num_images, \
            f"Expected {num_images} images, got {len(images)}"
        
        # Verify each image is a numpy array
        for i, img in enumerate(images):
            assert isinstance(img, np.ndarray), \
                f"Image {i} should be a numpy array"
            
            # Verify image shape (HWC format)
            assert len(img.shape) == 3, \
                f"Image {i} should have 3 dimensions (H, W, C)"
            
            # Verify image has 3 channels (RGB)
            assert img.shape[2] == 3, \
                f"Image {i} should have 3 channels (RGB)"
            
            # Verify image values are in uint8 range [0, 255]
            assert img.dtype == np.uint8, \
                f"Image {i} should be uint8 type"
            assert np.all(img >= 0) and np.all(img <= 255), \
                f"Image {i} values should be in range [0, 255]"
    
    @given(
        device=st.sampled_from(["cpu"])
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_12_synthetic_image_generation_without_model(self, device):
        """
        Property 12: Synthetic Image Generation (Error Handling)
        
        Verify that generating images without loading a model raises an error.
        
        **Validates: Requirements 6.2**
        """
        inference = ConSinGANInference(self.test_session_id, device=device)
        
        # Try to generate without loading model
        with pytest.raises(RuntimeError):
            inference.generate(num_images=5)
    
    @given(
        num_images=st.integers(min_value=1, max_value=5),
        device=st.sampled_from(["cpu"])
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_12_synthetic_image_generation_consistency(self, num_images, device):
        """
        Property 12: Synthetic Image Generation (Consistency)
        
        Verify that generated images have consistent properties.
        
        **Validates: Requirements 6.2**
        """
        inference = ConSinGANInference(self.test_session_id, device=device)
        inference.load_model(self.test_session_id)
        
        # Generate images
        images = inference.generate(num_images=num_images)
        
        # Verify all images have same shape
        first_shape = images[0].shape
        for i, img in enumerate(images):
            assert img.shape == first_shape, \
                f"Image {i} shape {img.shape} doesn't match first image shape {first_shape}"


class TestConSinGANInferenceBase64Encoding:
    """Property tests for base64 image encoding."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Create a temporary session for testing
        self.test_session_id = SessionManager.create_session()
        
        # Create and save dummy models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        torch.save(generator, os.path.join(models_path, "generator.pth"))
        torch.save(discriminator, os.path.join(models_path, "discriminator.pth"))
        
        yield
        
        # Cleanup
        uploads_path = SessionManager.get_session_path(self.test_session_id, "uploads")
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    @given(
        num_images=st.integers(min_value=1, max_value=5),
        device=st.sampled_from(["cpu"])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_13_base64_image_encoding(self, num_images, device):
        """
        Property 13: Base64 Image Encoding
        
        For any generated image, it should be encoded in base64 format and
        returned with "Sentetik (AI)" metadata label.
        
        **Validates: Requirements 6.3, 6.4**
        """
        inference = ConSinGANInference(self.test_session_id, device=device)
        inference.load_model(self.test_session_id)
        
        # Generate base64 encoded images
        result = inference.generate_base64(num_images=num_images)
        
        # Verify correct number of results
        assert len(result) == num_images, \
            f"Expected {num_images} results, got {len(result)}"
        
        # Verify each result has correct structure
        for i, item in enumerate(result):
            assert isinstance(item, dict), \
                f"Result {i} should be a dictionary"
            
            # Verify required keys
            assert 'image' in item, f"Result {i} missing 'image' key"
            assert 'label' in item, f"Result {i} missing 'label' key"
            assert 'ssim' in item, f"Result {i} missing 'ssim' key"
            
            # Verify image is base64 encoded data URI
            assert item['image'].startswith('data:image/jpeg;base64,'), \
                f"Result {i} image should be a JPEG data URI"
            
            # Verify label is correct
            assert item['label'] == 'Sentetik (AI)', \
                f"Result {i} label should be 'Sentetik (AI)', got {item['label']}"
            
            # Verify SSIM is a number
            assert isinstance(item['ssim'], (int, float)), \
                f"Result {i} SSIM should be a number"
    
    @given(
        num_images=st.integers(min_value=1, max_value=3),
        device=st.sampled_from(["cpu"])
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_13_base64_image_encoding_decodable(self, num_images, device):
        """
        Property 13: Base64 Image Encoding (Decodability)
        
        Verify that base64 encoded images can be decoded back to valid images.
        
        **Validates: Requirements 6.3, 6.4**
        """
        import base64
        from PIL import Image
        from io import BytesIO
        
        inference = ConSinGANInference(self.test_session_id, device=device)
        inference.load_model(self.test_session_id)
        
        # Generate base64 encoded images
        result = inference.generate_base64(num_images=num_images)
        
        # Verify each image can be decoded
        for i, item in enumerate(result):
            # Extract base64 string
            data_uri = item['image']
            base64_str = data_uri.replace('data:image/jpeg;base64,', '')
            
            # Decode base64
            try:
                image_data = base64.b64decode(base64_str)
                image = Image.open(BytesIO(image_data))
                
                # Verify image properties
                assert image.format == 'JPEG', f"Result {i} should be JPEG format"
                assert image.mode == 'RGB', f"Result {i} should be RGB mode"
                assert image.size[0] > 0 and image.size[1] > 0, \
                    f"Result {i} should have valid dimensions"
            except Exception as e:
                pytest.fail(f"Failed to decode base64 image {i}: {e}")


class TestConSinGANInferenceIntegration:
    """Integration tests for inference module."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Create a temporary session for testing
        self.test_session_id = SessionManager.create_session()
        
        # Create and save dummy models
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        torch.save(generator, os.path.join(models_path, "generator.pth"))
        torch.save(discriminator, os.path.join(models_path, "discriminator.pth"))
        
        yield
        
        # Cleanup
        uploads_path = SessionManager.get_session_path(self.test_session_id, "uploads")
        models_path = SessionManager.get_session_path(self.test_session_id, "models")
        
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
    
    @given(
        device=st.sampled_from(["cpu"])
    )
    @settings(max_examples=3, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_inference_end_to_end(self, device):
        """
        End-to-end test: Load model -> Generate images -> Encode as base64
        
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        # Initialize inference
        inference = ConSinGANInference(self.test_session_id, device=device)
        
        # Load model
        assert inference.load_model(self.test_session_id) is True
        
        # Generate images
        images = inference.generate(num_images=3)
        assert len(images) == 3
        
        # Generate base64 encoded images
        result = inference.generate_base64(num_images=3)
        assert len(result) == 3
        
        # Verify all results are valid
        for item in result:
            assert 'image' in item
            assert 'label' in item
            assert item['label'] == 'Sentetik (AI)'
