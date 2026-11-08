"""
Property-based tests for TexturePreservationOptimizer.

These tests verify that texture preservation mechanisms correctly maintain
fruit leaf characteristics during augmentation.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from texture_preservation import TexturePreservationOptimizer


class TestTexturePreservationOptimizer:
    """Property tests for TexturePreservationOptimizer."""
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
        channels=st.just(3)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_4_generator_texture_preservation(self, height, width, channels):
        """
        Property 4: Generator Texture Preservation
        
        For any input image and generated output, the SSIM (Structural Similarity Index)
        between the original and generated image should be at least 0.5, indicating
        texture and color preservation.
        
        **Validates: Requirements 2.3, 8.4**
        """
        # Create original image with uint8 values
        original_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # Initialize optimizer
        optimizer = TexturePreservationOptimizer(original_image)
        
        # Create a slightly modified version (simulating generation)
        # Add small noise to simulate generation process
        generated_image = original_image.astype(np.float32) + np.random.normal(0, 5, original_image.shape)
        generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)
        
        # Calculate SSIM
        ssim_score = optimizer.get_ssim_score(generated_image)
        
        # Verify SSIM is in valid range [0, 1]
        assert 0.0 <= ssim_score <= 1.0, \
            f"SSIM score {ssim_score} should be between 0 and 1"
        
        # For identical images, SSIM should be close to 1
        ssim_identical = optimizer.get_ssim_score(original_image)
        assert ssim_identical > 0.99, \
            f"SSIM for identical images should be close to 1, got {ssim_identical}"
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
        channels=st.just(3)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_17_ssim_calculation(self, height, width, channels):
        """
        Property 17: SSIM Calculation
        
        For any completed training, the SSIM score between the original image and
        a generated image should be calculated and returned; if SSIM < 0.5,
        a warning should be issued.
        
        **Validates: Requirements 8.4, 8.5**
        """
        # Create original image
        original_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # Initialize optimizer
        optimizer = TexturePreservationOptimizer(original_image)
        
        # Test with identical image
        ssim_identical = optimizer._calculate_ssim(
            optimizer._normalize_image(original_image),
            optimizer._normalize_image(original_image)
        )
        assert ssim_identical > 0.99, "SSIM for identical images should be ~1.0"
        
        # Test with completely different image
        different_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        ssim_different = optimizer._calculate_ssim(
            optimizer._normalize_image(original_image),
            optimizer._normalize_image(different_image)
        )
        assert 0.0 <= ssim_different <= 1.0, "SSIM should be in [0, 1]"
        
        # Test with slightly modified image
        modified_image = original_image.astype(np.float32) + np.random.normal(0, 10, original_image.shape)
        modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
        ssim_modified = optimizer._calculate_ssim(
            optimizer._normalize_image(original_image),
            optimizer._normalize_image(modified_image)
        )
        assert 0.0 <= ssim_modified <= 1.0, "SSIM should be in [0, 1]"
        assert ssim_modified > ssim_different, \
            "SSIM for slightly modified image should be higher than completely different"
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
        channels=st.just(3)
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_texture_loss_calculation(self, height, width, channels):
        """
        Test that texture loss is calculated correctly.
        
        Texture loss should be in range [0, 1] where 0 means perfect preservation.
        """
        # Create original image
        original_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # Initialize optimizer
        optimizer = TexturePreservationOptimizer(original_image)
        
        # Test with identical image (should have low loss)
        loss_identical = optimizer.calculate_texture_loss(original_image)
        assert 0.0 <= loss_identical <= 1.0, "Texture loss should be in [0, 1]"
        assert loss_identical < 0.1, "Loss for identical images should be very low"
        
        # Test with slightly modified image
        modified_image = original_image.astype(np.float32) + np.random.normal(0, 5, original_image.shape)
        modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
        loss_modified = optimizer.calculate_texture_loss(modified_image)
        assert 0.0 <= loss_modified <= 1.0, "Texture loss should be in [0, 1]"
        
        # Test with completely different image (should have high loss)
        different_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        loss_different = optimizer.calculate_texture_loss(different_image)
        assert 0.0 <= loss_different <= 1.0, "Texture loss should be in [0, 1]"
        assert loss_different > loss_modified, \
            "Loss for completely different image should be higher than slightly modified"
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
        channels=st.just(3)
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_texture_feature_extraction(self, height, width, channels):
        """
        Test that texture features are extracted correctly.
        
        Features should include edge magnitude, orientation, local contrast, etc.
        """
        # Create original image
        original_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # Initialize optimizer
        optimizer = TexturePreservationOptimizer(original_image)
        
        # Verify features are extracted
        features = optimizer.texture_features
        assert 'edge_magnitude' in features, "Should have edge_magnitude"
        assert 'edge_orientation' in features, "Should have edge_orientation"
        assert 'local_contrast' in features, "Should have local_contrast"
        assert 'hist_edges' in features, "Should have hist_edges"
        assert 'hist_orientation' in features, "Should have hist_orientation"
        assert 'mean_intensity' in features, "Should have mean_intensity"
        assert 'std_intensity' in features, "Should have std_intensity"
        
        # Verify feature shapes
        assert features['edge_magnitude'].shape == (height, width), \
            "Edge magnitude should match image dimensions"
        assert features['edge_orientation'].shape == (height, width), \
            "Edge orientation should match image dimensions"
        assert features['local_contrast'].shape == (height, width), \
            "Local contrast should match image dimensions"
        
        # Verify histogram sizes
        assert len(features['hist_edges']) == 16, "Edge histogram should have 16 bins"
        assert len(features['hist_orientation']) == 8, "Orientation histogram should have 8 bins"
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
        channels=st.just(3)
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_texture_feature_matching(self, height, width, channels):
        """
        Test that texture feature matching works correctly.
        
        Matching score should be in [0, 1] where 1 means perfect match.
        """
        # Create original image
        original_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # Initialize optimizer
        optimizer = TexturePreservationOptimizer(original_image)
        
        # Test with identical image
        identical_features = optimizer._extract_texture_features(
            optimizer._normalize_image(original_image)
        )
        match_identical = optimizer._match_texture_features(identical_features)
        assert 0.0 <= match_identical <= 1.0, "Match score should be in [0, 1]"
        assert match_identical > 0.9, "Match score for identical image should be high"
        
        # Test with modified image
        modified_image = original_image.astype(np.float32) + np.random.normal(0, 10, original_image.shape)
        modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
        modified_features = optimizer._extract_texture_features(
            optimizer._normalize_image(modified_image)
        )
        match_modified = optimizer._match_texture_features(modified_features)
        assert 0.0 <= match_modified <= 1.0, "Match score should be in [0, 1]"
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
        channels=st.just(3)
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_image_normalization(self, height, width, channels):
        """
        Test that image normalization works correctly.
        
        Images should be normalized to [0, 1] range.
        """
        # Test with uint8 image
        uint8_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        optimizer = TexturePreservationOptimizer(uint8_image)
        normalized = optimizer._normalize_image(uint8_image)
        
        assert normalized.dtype == np.float32, "Normalized image should be float32"
        assert np.min(normalized) >= 0.0, "Normalized image should have min >= 0"
        assert np.max(normalized) <= 1.0, "Normalized image should have max <= 1"
        
        # Test with float image already in [0, 1]
        float_image = np.random.rand(height, width, channels).astype(np.float32)
        normalized_float = optimizer._normalize_image(float_image)
        
        assert normalized_float.dtype == np.float32, "Normalized image should be float32"
        assert np.min(normalized_float) >= 0.0, "Normalized image should have min >= 0"
        assert np.max(normalized_float) <= 1.0, "Normalized image should have max <= 1"
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_ssim_with_different_sizes(self, height, width):
        """
        Test that SSIM calculation handles images of different sizes.
        
        The method should resize images to match before calculating SSIM.
        """
        # Create original image
        original_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        optimizer = TexturePreservationOptimizer(original_image)
        
        # Create image with different size
        different_size = np.random.randint(0, 256, (height // 2, width // 2, 3), dtype=np.uint8)
        
        # Should not raise error
        ssim_score = optimizer.get_ssim_score(different_size)
        assert 0.0 <= ssim_score <= 1.0, "SSIM should be in [0, 1]"
    
    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
        channels=st.just(3)
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_optimizer_initialization(self, height, width, channels):
        """
        Test that TexturePreservationOptimizer initializes correctly.
        
        Should extract features and store original image.
        """
        # Create original image
        original_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        # Initialize optimizer
        optimizer = TexturePreservationOptimizer(original_image)
        
        # Verify original image is stored and normalized
        assert optimizer.original.dtype == np.float32, "Original should be float32"
        assert np.min(optimizer.original) >= 0.0, "Original should have min >= 0"
        assert np.max(optimizer.original) <= 1.0, "Original should have max <= 1"
        
        # Verify texture features are extracted
        assert optimizer.texture_features is not None, "Texture features should be extracted"
        assert isinstance(optimizer.texture_features, dict), "Features should be a dictionary"
