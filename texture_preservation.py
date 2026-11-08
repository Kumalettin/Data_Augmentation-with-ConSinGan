"""
Texture Preservation Module for ConSinGAN

This module ensures that fruit leaf textures and disease characteristics
are preserved during augmentation using SSIM and texture feature matching.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage


class TexturePreservationOptimizer:
    """
    Ensures that fruit leaf textures and disease characteristics are preserved
    during augmentation.
    
    Uses SSIM (Structural Similarity Index) and texture feature extraction
    to maintain visual quality and texture characteristics.
    """
    
    def __init__(self, original_image):
        """
        Initialize the TexturePreservationOptimizer with an original image.
        
        Args:
            original_image: Original fruit leaf image as numpy array (H, W, C)
                           Values should be in range [0, 255] or [0, 1]
        """
        self.original = self._normalize_image(original_image)
        self.texture_features = self._extract_texture_features(self.original)
    
    def _normalize_image(self, image):
        """
        Normalize image to [0, 1] range if needed.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image in [0, 1] range
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    
    def _extract_texture_features(self, image):
        """
        Extract texture features using edge detection and local patterns.
        
        This method extracts texture characteristics including:
        - Edge map (Sobel filter)
        - Local binary patterns
        - Texture orientation
        
        Args:
            image: Input image as numpy array (H, W, C) in [0, 1] range
            
        Returns:
            Dictionary containing extracted texture features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Compute Sobel gradients for edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute edge magnitude and orientation
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_orientation = np.arctan2(sobel_y, sobel_x)
        
        # Compute local contrast (texture roughness)
        local_mean = ndimage.uniform_filter(gray.astype(np.float32), size=5)
        local_variance = ndimage.uniform_filter(
            (gray.astype(np.float32) - local_mean)**2, size=5
        )
        local_contrast = np.sqrt(local_variance)
        
        # Compute histogram of oriented gradients (simplified)
        hist_edges = np.histogram(edge_magnitude.flatten(), bins=16)[0]
        hist_orientation = np.histogram(edge_orientation.flatten(), bins=8)[0]
        
        features = {
            'edge_magnitude': edge_magnitude,
            'edge_orientation': edge_orientation,
            'local_contrast': local_contrast,
            'hist_edges': hist_edges,
            'hist_orientation': hist_orientation,
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray)
        }
        
        return features
    
    def calculate_texture_loss(self, generated_image):
        """
        Calculate loss based on texture similarity between original and generated images.
        
        Uses SSIM and texture feature matching to ensure generated images
        maintain the characteristics of the original fruit leaf image.
        
        Args:
            generated_image: Generated image as numpy array (H, W, C)
                            Values should be in range [0, 255] or [0, 1]
            
        Returns:
            Texture loss value in range [0, 1] where 0 means perfect preservation
        """
        generated = self._normalize_image(generated_image)
        
        # Calculate SSIM
        ssim_score = self._calculate_ssim(self.original, generated)
        
        # Calculate texture feature matching
        gen_features = self._extract_texture_features(generated)
        texture_match = self._match_texture_features(gen_features)
        
        # Weighted combination: SSIM (70%) + texture features (30%)
        texture_loss = (1 - ssim_score) * 0.7 + (1 - texture_match) * 0.3
        
        return float(texture_loss)
    
    def _calculate_ssim(self, img1, img2):
        """
        Calculate Structural Similarity Index (SSIM) between two images.
        
        SSIM measures the perceived quality of images and is more aligned
        with human visual perception than MSE.
        
        Args:
            img1: First image as numpy array in [0, 1] range
            img2: Second image as numpy array in [0, 1] range
            
        Returns:
            SSIM score in range [0, 1] where 1 means identical images
        """
        # Ensure images have the same shape
        if img1.shape != img2.shape:
            # Resize img2 to match img1
            if len(img2.shape) == 3:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            else:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Handle both grayscale and color images
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            # For color images, compute SSIM for each channel and average
            ssim_score = ssim(img1, img2, channel_axis=2, data_range=1.0)
        else:
            # For grayscale images
            ssim_score = ssim(img1, img2, data_range=1.0)
        
        # Ensure score is in [0, 1] range
        return float(np.clip(ssim_score, 0, 1))
    
    def _match_texture_features(self, generated_features):
        """
        Match texture features between original and generated images.
        
        Compares extracted texture features and returns a similarity score.
        
        Args:
            generated_features: Dictionary of texture features from generated image
            
        Returns:
            Feature matching score in range [0, 1] where 1 means perfect match
        """
        # Compare edge magnitude distributions
        orig_edge_mean = np.mean(self.texture_features['edge_magnitude'])
        gen_edge_mean = np.mean(generated_features['edge_magnitude'])
        edge_similarity = 1 - np.abs(orig_edge_mean - gen_edge_mean) / (orig_edge_mean + 1e-6)
        
        # Compare local contrast
        orig_contrast_mean = np.mean(self.texture_features['local_contrast'])
        gen_contrast_mean = np.mean(generated_features['local_contrast'])
        contrast_similarity = 1 - np.abs(orig_contrast_mean - gen_contrast_mean) / (orig_contrast_mean + 1e-6)
        
        # Compare intensity statistics
        orig_mean = self.texture_features['mean_intensity']
        gen_mean = generated_features['mean_intensity']
        intensity_similarity = 1 - np.abs(orig_mean - gen_mean) / 255.0
        
        # Compare histogram of edges
        orig_hist = self.texture_features['hist_edges']
        gen_hist = generated_features['hist_edges']
        # Normalize histograms
        orig_hist = orig_hist / (np.sum(orig_hist) + 1e-6)
        gen_hist = gen_hist / (np.sum(gen_hist) + 1e-6)
        # Chi-square distance
        hist_distance = np.sum((orig_hist - gen_hist)**2 / (orig_hist + gen_hist + 1e-6))
        hist_similarity = 1 / (1 + hist_distance)
        
        # Weighted combination of all similarities
        overall_similarity = (
            edge_similarity * 0.3 +
            contrast_similarity * 0.3 +
            intensity_similarity * 0.2 +
            hist_similarity * 0.2
        )
        
        return float(np.clip(overall_similarity, 0, 1))
    
    def get_ssim_score(self, generated_image):
        """
        Get SSIM score between original and generated image.
        
        Convenience method to get just the SSIM score without full texture loss.
        
        Args:
            generated_image: Generated image as numpy array
            
        Returns:
            SSIM score in range [0, 1]
        """
        generated = self._normalize_image(generated_image)
        return self._calculate_ssim(self.original, generated)
