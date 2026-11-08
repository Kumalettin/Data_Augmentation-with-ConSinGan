"""
Property-based tests for augmentation module.
Tests format-agnostic, source-agnostic, and metadata preservation properties.
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from PIL import Image
import cv2
import io

from augmentation import (
    apply_augmentations,
    apply_traditional_augmentations,
    ImageMetadata,
    detect_image_format,
    normalize_to_opencv,
    normalize_to_pil
)


# Strategies for generating test data
@st.composite
def opencv_images(draw):
    """Generate random OpenCV (BGR) images as numpy arrays"""
    height = draw(st.integers(min_value=32, max_value=256))
    width = draw(st.integers(min_value=32, max_value=256))
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


@st.composite
def pil_images(draw):
    """Generate random PIL (RGB) images"""
    height = draw(st.integers(min_value=32, max_value=256))
    width = draw(st.integers(min_value=32, max_value=256))
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, mode='RGB')


@st.composite
def augmentation_techniques(draw):
    """Generate lists of augmentation techniques"""
    all_techniques = ['rotate', 'flip', 'brightness', 'contrast', 'blur', 'noise']
    return draw(st.lists(
        st.sampled_from(all_techniques),
        min_size=1,
        max_size=6,
        unique=True
    ))


class TestFormatAgnosticAugmentation:
    """
    Property 18: Format-Agnostic Augmentation
    
    For any augmentation operation, the apply_augmentations() function should work
    with both OpenCV (BGR) and PIL (RGB) formatted images without requiring format
    conversion from the caller.
    
    Validates: Requirements 9.1, 9.3
    """
    
    @given(
        image_cv=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_opencv_format_augmentation(self, image_cv, techniques):
        """Test augmentation works with OpenCV (BGR) format"""
        result_cv, applied, metadata = apply_augmentations(image_cv, techniques)
        
        # Result should be numpy array
        assert isinstance(result_cv, np.ndarray)
        # Result should have same shape as input
        assert result_cv.shape == image_cv.shape
        # Result should be uint8
        assert result_cv.dtype == np.uint8
        # Applied techniques should be non-empty if techniques were provided
        if techniques:
            assert len(applied) > 0
    
    @given(
        image_pil=pil_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_pil_format_augmentation(self, image_pil, techniques):
        """Test augmentation works with PIL (RGB) format"""
        result_cv, applied, metadata = apply_augmentations(image_pil, techniques)
        
        # Result should be numpy array (OpenCV format)
        assert isinstance(result_cv, np.ndarray)
        # Result should have same shape as input
        assert result_cv.shape == (image_pil.height, image_pil.width, 3)
        # Result should be uint8
        assert result_cv.dtype == np.uint8
        # Applied techniques should be non-empty if techniques were provided
        if techniques:
            assert len(applied) > 0
    
    @given(
        image_cv=opencv_images(),
        image_pil=pil_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=30)
    def test_format_consistency(self, image_cv, image_pil, techniques):
        """Test that both formats produce consistent output shapes"""
        result_cv_from_cv, _, _ = apply_augmentations(image_cv, techniques)
        result_cv_from_pil, _, _ = apply_augmentations(image_pil, techniques)
        
        # Both should produce numpy arrays
        assert isinstance(result_cv_from_cv, np.ndarray)
        assert isinstance(result_cv_from_pil, np.ndarray)
        # Both should be uint8
        assert result_cv_from_cv.dtype == np.uint8
        assert result_cv_from_pil.dtype == np.uint8


class TestSourceAgnosticAugmentation:
    """
    Property 19: Source-Agnostic Augmentation
    
    For any augmentation technique, it should produce consistent results regardless
    of whether the input image is original or AI-generated, as long as the image
    is in the correct format.
    
    Validates: Requirements 9.2
    """
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_original_image_augmentation(self, image, techniques):
        """Test augmentation on original images"""
        metadata = ImageMetadata(source="original")
        result, applied, meta = apply_augmentations(image, techniques, metadata)
        
        # Should preserve source metadata
        assert meta.source == "original"
        # Should have valid output
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_synthetic_image_augmentation(self, image, techniques):
        """Test augmentation on AI-generated (synthetic) images"""
        metadata = ImageMetadata(source="synthetic")
        result, applied, meta = apply_augmentations(image, techniques, metadata)
        
        # Should preserve source metadata
        assert meta.source == "synthetic"
        # Should have valid output
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_traditional_augmented_image(self, image, techniques):
        """Test augmentation on traditionally augmented images"""
        metadata = ImageMetadata(source="traditional")
        result, applied, meta = apply_augmentations(image, techniques, metadata)
        
        # Should preserve source metadata
        assert meta.source == "traditional"
        # Should have valid output
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape


class TestMetadataPreservation:
    """
    Property 20: Metadata Preservation
    
    For any augmented image, the metadata (source, technique, timestamp) should be
    preserved and returned with the image.
    
    Validates: Requirements 9.4
    """
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_metadata_preservation_with_source(self, image, techniques):
        """Test that source metadata is preserved"""
        metadata = ImageMetadata(source="original")
        result, applied, returned_meta = apply_augmentations(image, techniques, metadata)
        
        # Source should be preserved
        assert returned_meta.source == "original"
        # Metadata should be returned
        assert returned_meta is not None
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_metadata_preservation_with_ssim(self, image, techniques):
        """Test that SSIM score metadata is preserved"""
        metadata = ImageMetadata(source="synthetic", ssim_score=0.75)
        result, applied, returned_meta = apply_augmentations(image, techniques, metadata)
        
        # SSIM should be preserved
        assert returned_meta.ssim_score == 0.75
        # Source should be preserved
        assert returned_meta.source == "synthetic"
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_technique_metadata_updated(self, image, techniques):
        """Test that applied techniques are recorded in metadata"""
        metadata = ImageMetadata(source="original")
        result, applied, returned_meta = apply_augmentations(image, techniques, metadata)
        
        # If techniques were applied, metadata should be updated
        if applied:
            assert returned_meta.technique is not None
            # Technique string should contain applied techniques
            assert len(returned_meta.technique) > 0
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=50)
    def test_metadata_none_handling(self, image, techniques):
        """Test that None metadata is handled gracefully"""
        result, applied, returned_meta = apply_augmentations(image, techniques, metadata=None)
        
        # Should return valid metadata even if None was passed
        assert returned_meta is not None
        assert isinstance(returned_meta, ImageMetadata)
        # Should have default source
        assert returned_meta.source == "unknown"


class TestBackwardCompatibility:
    """Test backward compatibility with legacy augmentation function"""
    
    @given(
        image=opencv_images(),
        techniques=augmentation_techniques()
    )
    @settings(max_examples=30)
    def test_legacy_function_compatibility(self, image, techniques):
        """Test that legacy apply_traditional_augmentations still works"""
        result, applied = apply_traditional_augmentations(image, techniques)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
        # Should have same shape
        assert result.shape == image.shape
        # Should return applied techniques list
        assert isinstance(applied, list)


class TestFormatDetection:
    """Test image format detection"""
    
    @given(image=opencv_images())
    @settings(max_examples=20)
    def test_detect_opencv_format(self, image):
        """Test detection of OpenCV format"""
        fmt = detect_image_format(image)
        assert fmt == "opencv"
    
    @given(image=pil_images())
    @settings(max_examples=20)
    def test_detect_pil_format(self, image):
        """Test detection of PIL format"""
        fmt = detect_image_format(image)
        assert fmt == "pil"


class TestFormatNormalization:
    """Test format normalization functions"""
    
    @given(image=opencv_images())
    @settings(max_examples=20)
    def test_normalize_opencv_to_pil(self, image):
        """Test converting OpenCV to PIL"""
        pil_img = normalize_to_pil(image)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (image.shape[1], image.shape[0])
    
    @given(image=pil_images())
    @settings(max_examples=20)
    def test_normalize_pil_to_opencv(self, image):
        """Test converting PIL to OpenCV"""
        cv_img = normalize_to_opencv(image)
        assert isinstance(cv_img, np.ndarray)
        assert cv_img.shape == (image.height, image.width, 3)
    
    @given(image=opencv_images())
    @settings(max_examples=20)
    def test_normalize_opencv_idempotent(self, image):
        """Test that normalizing OpenCV to OpenCV is idempotent"""
        result = normalize_to_opencv(image)
        assert np.array_equal(result, image)
    
    @given(image=pil_images())
    @settings(max_examples=20)
    def test_normalize_pil_idempotent(self, image):
        """Test that normalizing PIL to PIL is idempotent"""
        result = normalize_to_pil(image)
        assert result == image
