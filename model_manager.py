"""
Model Manager for ConSinGAN

Manages trained model storage, retrieval, and organization with:
- Model library with date/name based organization
- Reference image storage
- Generated image organization by date
- Model metadata tracking
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages trained ConSinGAN models with persistent storage and retrieval.
    """
    
    def __init__(self, base_path: str = "model_library"):
        """
        Initialize the Model Manager.
        
        Args:
            base_path: Base directory for model library (default: "model_library")
        """
        self.base_path = base_path
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelManager initialized with base path: {base_path}")
    
    def save_trained_model(
        self,
        model_name: str,
        generator_state: Dict,
        discriminator_state: Dict,
        reference_image_path: str,
        config: Dict[str, Any],
        description: str = ""
    ) -> str:
        """
        Save a trained model to the library.
        
        Args:
            model_name: Name for the model (e.g., "apple_leaf_v1")
            generator_state: Generator model state dict
            discriminator_state: Discriminator model state dict
            reference_image_path: Path to the reference/training image
            config: Training configuration
            description: Optional description of the model
        
        Returns:
            Model ID (date_time_name format)
        """
        try:
            # Create model ID with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{timestamp}_{model_name}"
            
            # Create model directory
            model_dir = os.path.join(self.base_path, model_id)
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            
            # Save model states
            import torch
            torch.save(generator_state, os.path.join(model_dir, "generator.pth"))
            torch.save(discriminator_state, os.path.join(model_dir, "discriminator.pth"))
            
            # Copy reference image
            if os.path.exists(reference_image_path):
                ref_dest = os.path.join(model_dir, "reference_image.jpg")
                shutil.copy2(reference_image_path, ref_dest)
            
            # Save metadata
            metadata = {
                'model_id': model_id,
                'model_name': model_name,
                'created_date': datetime.now().isoformat(),
                'description': description,
                'config': config,
                'reference_image': 'reference_image.jpg'
            }
            
            with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved: {model_id}")
            return model_id
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available trained models.
        
        Returns:
            List of model metadata dictionaries
        """
        try:
            models = []
            
            for model_dir in sorted(os.listdir(self.base_path), reverse=True):
                model_path = os.path.join(self.base_path, model_dir)
                
                if not os.path.isdir(model_path):
                    continue
                
                metadata_path = os.path.join(model_path, "metadata.json")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add path info
                    metadata['path'] = model_path
                    models.append(metadata)
            
            logger.info(f"Found {len(models)} trained models")
            return models
        
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific model by ID.
        
        Args:
            model_id: Model ID to retrieve
        
        Returns:
            Model metadata and paths, or None if not found
        """
        try:
            model_path = os.path.join(self.base_path, model_id)
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_id}")
                return None
            
            metadata_path = os.path.join(model_path, "metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata not found for model: {model_id}")
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add file paths
            metadata['path'] = model_path
            metadata['generator_path'] = os.path.join(model_path, "generator.pth")
            metadata['discriminator_path'] = os.path.join(model_path, "discriminator.pth")
            metadata['reference_image_path'] = os.path.join(model_path, "reference_image.jpg")
            
            return metadata
        
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    def create_generation_folder(self, model_id: str) -> str:
        """
        Create a dated folder for generated images from a model.
        
        Args:
            model_id: Model ID
        
        Returns:
            Path to the generation folder
        """
        try:
            model_path = os.path.join(self.base_path, model_id)
            
            # Create generations folder if it doesn't exist
            generations_path = os.path.join(model_path, "generations")
            Path(generations_path).mkdir(parents=True, exist_ok=True)
            
            # Create dated subfolder
            date_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            dated_path = os.path.join(generations_path, date_folder)
            Path(dated_path).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created generation folder: {dated_path}")
            return dated_path
        
        except Exception as e:
            logger.error(f"Failed to create generation folder: {e}")
            raise
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the library.
        
        Args:
            model_id: Model ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = os.path.join(self.base_path, model_id)
            
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                logger.info(f"Model deleted: {model_id}")
                return True
            
            logger.warning(f"Model not found for deletion: {model_id}")
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def rename_model(self, model_id: str, new_name: str) -> Optional[str]:
        """
        Rename a model.
        
        Args:
            model_id: Current model ID
            new_name: New name for the model
        
        Returns:
            New model ID, or None if failed
        """
        try:
            model_data = self.get_model(model_id)
            
            if not model_data:
                logger.warning(f"Model not found: {model_id}")
                return None
            
            # Create new model ID with new name
            timestamp = model_id.split('_')[0] + '_' + model_id.split('_')[1]
            new_model_id = f"{timestamp}_{new_name}"
            
            # Rename directory
            old_path = os.path.join(self.base_path, model_id)
            new_path = os.path.join(self.base_path, new_model_id)
            
            os.rename(old_path, new_path)
            
            # Update metadata
            metadata_path = os.path.join(new_path, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['model_id'] = new_model_id
            metadata['model_name'] = new_name
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model renamed: {model_id} -> {new_model_id}")
            return new_model_id
        
        except Exception as e:
            logger.error(f"Failed to rename model: {e}")
            return None
