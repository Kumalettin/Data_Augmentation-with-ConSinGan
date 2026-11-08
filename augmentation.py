"""
Format-agnostic augmentation module supporting both OpenCV and PIL formats.
Preserves metadata and works with both original and AI-generated images.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
from typing import Tuple, List, Dict, Optional, Union


class ImageMetadata:
    """Stores metadata about an image"""
    def __init__(self, source: str, technique: Optional[str] = None, ssim_score: Optional[float] = None):
        self.source = source  # "original", "traditional", "synthetic"
        self.technique = technique
        self.ssim_score = ssim_score


def detect_image_format(image: Union[np.ndarray, Image.Image]) -> str:
    """
    Detect if image is in OpenCV (BGR) or PIL (RGB) format.
    
    Args:
        image: Image as numpy array or PIL Image
        
    Returns:
        "opencv" for BGR numpy array, "pil" for PIL Image, "numpy_rgb" for RGB numpy array
    """
    if isinstance(image, Image.Image):
        return "pil"
    elif isinstance(image, np.ndarray):
        # Assume BGR if numpy array (OpenCV convention)
        return "opencv"
    else:
        raise ValueError(f"Unsupported image format: {type(image)}")


def normalize_to_opencv(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert any image format to OpenCV (BGR) numpy array.
    
    Args:
        image: Image in any supported format
        
    Returns:
        Image as BGR numpy array
    """
    if isinstance(image, Image.Image):
        # PIL RGB to OpenCV BGR
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError(f"Unsupported image format: {type(image)}")


def normalize_to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Convert any image format to PIL (RGB) Image.
    
    Args:
        image: Image in any supported format
        
    Returns:
        Image as PIL Image (RGB)
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        # Assume BGR numpy array, convert to RGB PIL
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Unsupported image format: {type(image)}")


def apply_augmentations(
    image: Union[np.ndarray, Image.Image],
    techniques: List[str],
    metadata: Optional[ImageMetadata] = None
) -> Tuple[np.ndarray, List[str], ImageMetadata]:
    """
    Apply augmentation techniques to an image in a format-agnostic way.
    
    Supports both OpenCV (BGR) and PIL (RGB) formats.
    Works with both original and AI-generated images.
    Preserves metadata throughout the process.
    
    Args:
        image: Image as numpy array (OpenCV BGR) or PIL Image (RGB)
        techniques: List of augmentation techniques to apply
        metadata: Optional metadata to preserve
        
    Returns:
        Tuple of (augmented_image_cv, applied_techniques_list, metadata)
    """
    # Normalize to OpenCV format for processing
    img_cv = normalize_to_opencv(image)
    img_pil = normalize_to_pil(image)
    
    # Initialize metadata if not provided
    if metadata is None:
        metadata = ImageMetadata(source="unknown")
    
    applied = []
    
    # Shuffle techniques for variety
    random.shuffle(techniques)
    
    # 70% chance to apply each technique (more variety)
    active_techniques = [t for t in techniques if random.random() > 0.3]
    
    # Ensure at least one technique if any were provided
    if not active_techniques and techniques:
        active_techniques = [random.choice(techniques)]
    
    for tech in active_techniques:
        if tech == 'rotate':
            angle = random.uniform(-45, 45)
            h, w = img_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_cv = cv2.warpAffine(img_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            img_pil = normalize_to_pil(img_cv)
            applied.append(f'🔄 Döndürme ({angle:.1f}°)')
            
        elif tech == 'flip':
            mode = random.choice([0, 1])  # 0: Vertical, 1: Horizontal
            img_cv = cv2.flip(img_cv, mode)
            img_pil = normalize_to_pil(img_cv)
            tag = "Yatay Aynalama" if mode == 1 else "Dikey Aynalama"
            applied.append(f'🪞 {tag}')
            
        elif tech == 'brightness':
            factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(factor)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            applied.append(f'☀️ Parlaklık ({factor:.2f}x)')
            
        elif tech == 'contrast':
            factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(factor)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            applied.append(f'🎨 Kontrast ({factor:.2f}x)')
            
        elif tech == 'blur':
            kernel_size = random.choice([3, 5, 7])
            img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
            img_pil = normalize_to_pil(img_cv)
            applied.append(f'🌫️ Bulanıklık ({kernel_size}x{kernel_size})')
            
        elif tech == 'noise':
            row, col, ch = img_cv.shape
            mean = 0
            var = random.uniform(10, 50)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch).astype(np.float32)
            
            noisy = img_cv.astype(np.float32) + gauss
            np.clip(noisy, 0, 255, out=noisy)
            img_cv = noisy.astype(np.uint8)
            
            img_pil = normalize_to_pil(img_cv)
            applied.append(f'📡 Gürültü')
    
    # Update metadata with applied techniques
    if applied:
        metadata.technique = ", ".join(applied)
    
    return img_cv, applied, metadata


def apply_traditional_augmentations(
    image: Union[np.ndarray, Image.Image],
    techniques: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Legacy function for backward compatibility.
    Applies traditional augmentations and returns OpenCV format.
    
    Args:
        image: Image in any supported format
        techniques: List of augmentation techniques
        
    Returns:
        Tuple of (augmented_image_cv, applied_techniques_list)
    """
    metadata = ImageMetadata(source="traditional")
    img_cv, applied, _ = apply_augmentations(image, techniques, metadata)
    return img_cv, applied
