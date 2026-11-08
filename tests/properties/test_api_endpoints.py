"""
Property-based tests for ConSinGAN API Endpoints.

These tests verify that the API endpoints maintain their correctness properties
across a wide range of inputs and configurations.
"""

import pytest
import torch
import numpy as np
import os
import shutil
import json
import base64
import time
from hypothesis import given, strategies as st, settings, HealthCheck
from pathlib import Path
from io import BytesIO
from PIL import Image

# Import Flask app and test client
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend import app, validate_training_config
from training import ConSinGANTrainer, TrainingTaskManager
from models import ConSinGANGenerator, ConSinGANDiscriminator
from device_manager import DeviceManager
from session_manager import SessionManager
from inference import ConSinGANInference


# Create test client with workaround for werkzeug version issue
app.config['TESTING'] = True

def get_test_client():
    """Get a test client, handling werkzeug version issues."""
    try:
        return app.test_client()
    except AttributeError:
        # Workaround for werkzeug version issue
        from werkzeug.test import Client
        return Client(app)

test_client = get_test_client()


@pytest.fixture
def session_id():
    """Create a test session."""
    session_id = SessionManager.create_session()
    yield session_id
    # Cleanup
    uploads_path = SessionManager.get_session_path(session_id, "uploads")
    models_path = SessionManager.get_session_path(session_id, "models")
    if os.path.exists(uploads_path):
        shutil.rmtree(uploads_path)
    if os.path.exists(models_path):
        shutil.rmtree(models_path)


@pytest.fixture
def trained_model(session_id):
    """Create a trained model for testing inference."""
    # Create and train a simple model
    generator = ConSinGANGenerator(num_scales=3)
    discriminator = ConSinGANDiscriminator(num_scales=3)
    device = "cpu"
    
    trainer = ConSinGANTrainer(generator, discriminator, device)
    
    # Save the model
    trainer.save_model(session_id)
    
    return session_id


class TestTrainingConfigValidation:
    """Property tests for training configuration validation."""
    
    @given(
        iterations=st.integers(min_value=50, max_value=15000),
        learning_rate=st.floats(min_value=0.000001, max_value=0.01),
        batch_size=st.integers(min_value=0, max_value=10),
        num_scales=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_10_hyperparameter_validation(self, iterations, learning_rate, batch_size, num_scales):
        """
        Property 10: Hyperparameter Validation
        
        For any training request with hyperparameters, invalid parameters
        (iterations < 100 or > 10000, learning_rate outside valid range)
        should be rejected with an error message, and defaults should be
        used if applicable.
        
        **Validates: Requirements 5.2, 5.3**
        """
        config = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_scales': num_scales
        }
        
        is_valid, error_msg = validate_training_config(config)
        
        # Check validity based on constraints
        expected_valid = (
            100 <= iterations <= 10000 and
            0.00001 <= learning_rate <= 0.001 and
            batch_size >= 1 and
            1 <= num_scales <= 5
        )
        
        assert is_valid == expected_valid, \
            f"Validation result {is_valid} doesn't match expected {expected_valid} for config {config}"
        
        if not is_valid:
            assert error_msg is not None, "Error message should be provided for invalid config"
            assert len(error_msg) > 0, "Error message should not be empty"


class TestTrainConsinganEndpoint:
    """Property tests for /api/train-consingan endpoint."""
    
    @given(
        iterations=st.sampled_from([100, 500, 1000, 5000, 10000]),
        learning_rate=st.sampled_from([0.00001, 0.0001, 0.0002, 0.001]),
        batch_size=st.sampled_from([1, 2]),
        num_scales=st.sampled_from([1, 2, 3])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None)
    def test_property_7_asynchronous_training_initiation(
        self, session_id, iterations, learning_rate, batch_size, num_scales
    ):
        """
        Property 7: Asynchronous Training Initiation
        
        For any POST request to /api/train-consingan, the endpoint should
        return immediately with a task ID without blocking, and training
        should proceed in the background.
        
        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        # Measure request time
        start_time = time.time()
        
        response = test_client.post('/api/train-consingan', json={
            'session_id': session_id,
            'iterations': iterations,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_scales': num_scales
        })
        
        request_time = time.time() - start_time
        
        # Verify response is successful
        assert response.status_code == 202, f"Expected 202, got {response.status_code}"
        
        data = response.get_json()
        assert data['success'] is True, "Response should indicate success"
        assert 'task_id' in data, "Response should contain task_id"
        assert len(data['task_id']) > 0, "task_id should not be empty"
        
        # Verify request returns quickly (< 1 second for async operation)
        assert request_time < 1.0, \
            f"Request took {request_time}s, should be < 1s for async operation"
    
    @given(
        iterations=st.sampled_from([50, 15000]),  # Invalid values
        learning_rate=st.sampled_from([0.000001, 0.01])  # Invalid values
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None)
    def test_property_10_hyperparameter_validation_endpoint(
        self, session_id, iterations, learning_rate
    ):
        """
        Property 10: Hyperparameter Validation (Endpoint)
        
        For any training request with invalid hyperparameters, the endpoint
        should reject the request with an error message.
        
        **Validates: Requirements 5.2, 5.3**
        """
        response = test_client.post('/api/train-consingan', json={
            'session_id': session_id,
            'iterations': iterations,
            'learning_rate': learning_rate
        })
        
        # Should reject invalid parameters
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        
        data = response.get_json()
        assert data['success'] is False, "Response should indicate failure"
        assert 'error' in data, "Response should contain error message"
    
    def test_missing_session_id(self):
        """Test that missing session_id is rejected."""
        response = test_client.post('/api/train-consingan', json={
            'iterations': 1000
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
    
    def test_invalid_session_id(self):
        """Test that invalid session_id is rejected."""
        response = test_client.post('/api/train-consingan', json={
            'session_id': 'invalid-session-id-that-does-not-exist',
            'iterations': 1000
        })
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False


class TestTrainingStatusEndpoint:
    """Property tests for /api/training-status/{task_id} endpoint."""
    
    def test_training_status_not_found(self):
        """Test that non-existent task returns 404."""
        response = test_client.get('/api/training-status/invalid-task-id')
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False
    
    def test_training_status_response_format(self, session_id):
        """Test that training status response has correct format."""
        # Start training
        train_response = test_client.post('/api/train-consingan', json={
            'session_id': session_id,
            'iterations': 100,
            'learning_rate': 0.0002
        })
        
        assert train_response.status_code == 202
        task_id = train_response.get_json()['task_id']
        
        # Get status
        status_response = test_client.get(f'/api/training-status/{task_id}')
        
        assert status_response.status_code == 200
        data = status_response.get_json()
        
        # Verify response format
        assert data['success'] is True
        assert 'status' in data
        assert 'progress' in data
        assert 'loss_g' in data
        assert 'loss_d' in data
        assert 'eta_seconds' in data
        
        # Verify data types
        assert isinstance(data['progress'], int)
        assert isinstance(data['loss_g'], (int, float))
        assert isinstance(data['loss_d'], (int, float))
        assert isinstance(data['eta_seconds'], (int, float))
        
        # Verify value ranges
        assert 0 <= data['progress'] <= 100
        assert data['loss_g'] >= 0
        assert data['loss_d'] >= 0
        assert data['eta_seconds'] >= 0


class TestGenerateConsinganEndpoint:
    """Property tests for /api/generate-consingan endpoint."""
    
    @given(
        num_images=st.sampled_from([1, 3, 5, 10])
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None)
    def test_property_12_synthetic_image_generation_count(
        self, trained_model, num_images
    ):
        """
        Property 12: Synthetic Image Generation Count
        
        For any inference request with num_images parameter, the system
        should generate exactly that many synthetic images (or return an
        error if impossible).
        
        **Validates: Requirements 6.2**
        """
        response = test_client.post('/api/generate-consingan', json={
            'session_id': trained_model,
            'num_images': num_images
        })
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['success'] is True
        assert 'images' in data
        assert len(data['images']) == num_images, \
            f"Expected {num_images} images, got {len(data['images'])}"
    
    @given(
        num_images=st.sampled_from([0, -1, 51, 100])  # Invalid values
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None)
    def test_invalid_num_images(self, trained_model, num_images):
        """Test that invalid num_images values are rejected."""
        response = test_client.post('/api/generate-consingan', json={
            'session_id': trained_model,
            'num_images': num_images
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
    
    def test_property_13_base64_image_encoding(self, trained_model):
        """
        Property 13: Base64 Image Encoding
        
        For any generated image, it should be encoded in base64 format
        and returned with "Sentetik (AI)" metadata label.
        
        **Validates: Requirements 6.3, 6.4**
        """
        response = test_client.post('/api/generate-consingan', json={
            'session_id': trained_model,
            'num_images': 3
        })
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['success'] is True
        assert 'images' in data
        
        for img_data in data['images']:
            # Verify image is base64 encoded
            assert 'image' in img_data
            assert img_data['image'].startswith('data:image/jpeg;base64,'), \
                "Image should be base64 encoded with proper data URI"
            
            # Verify label
            assert 'label' in img_data
            assert img_data['label'] == 'Sentetik (AI)', \
                "Image should have 'Sentetik (AI)' label"
            
            # Verify SSIM score exists
            assert 'ssim' in img_data
            assert isinstance(img_data['ssim'], (int, float))
            
            # Try to decode base64 to verify it's valid
            try:
                base64_str = img_data['image'].split(',')[1]
                decoded = base64.b64decode(base64_str)
                assert len(decoded) > 0, "Decoded image should not be empty"
            except Exception as e:
                pytest.fail(f"Failed to decode base64 image: {e}")
    
    def test_property_11_inference_model_loading(self, trained_model):
        """
        Property 11: Inference Model Loading
        
        For any inference request, if a trained model exists for the session,
        it should be loaded successfully; otherwise, an appropriate error
        should be returned.
        
        **Validates: Requirements 6.1, 6.5**
        """
        # Test with trained model (should succeed)
        response = test_client.post('/api/generate-consingan', json={
            'session_id': trained_model,
            'num_images': 1
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        
        # Test with non-existent model (should fail)
        response = test_client.post('/api/generate-consingan', json={
            'session_id': 'non-existent-session-id',
            'num_images': 1
        })
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False
    
    def test_generate_response_metadata(self, trained_model):
        """Test that generate response includes proper metadata."""
        response = test_client.post('/api/generate-consingan', json={
            'session_id': trained_model,
            'num_images': 2
        })
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Verify metadata
        assert 'metadata' in data
        metadata = data['metadata']
        
        assert 'model_version' in metadata
        assert 'generation_time' in metadata
        assert 'device' in metadata
        
        assert isinstance(metadata['generation_time'], (int, float))
        assert metadata['generation_time'] >= 0
        assert metadata['device'] in ['cuda', 'cpu']
    
    def test_missing_session_id(self):
        """Test that missing session_id is rejected."""
        response = test_client.post('/api/generate-consingan', json={
            'num_images': 5
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
    
    def test_invalid_session_id(self):
        """Test that invalid session_id is rejected."""
        response = test_client.post('/api/generate-consingan', json={
            'session_id': 'invalid-session-id-that-does-not-exist',
            'num_images': 5
        })
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False


class TestAPIEndpointIntegration:
    """Integration tests for API endpoints."""
    
    def test_end_to_end_training_and_inference(self, session_id):
        """Test complete workflow: train -> check status -> generate."""
        # Start training
        train_response = test_client.post('/api/train-consingan', json={
            'session_id': session_id,
            'iterations': 100,
            'learning_rate': 0.0002
        })
        
        assert train_response.status_code == 202
        task_id = train_response.get_json()['task_id']
        
        # Check training status
        status_response = test_client.get(f'/api/training-status/{task_id}')
        assert status_response.status_code == 200
        
        status_data = status_response.get_json()
        assert status_data['success'] is True
        assert 'status' in status_data
    
    def test_multiple_training_tasks(self):
        """Test that multiple training tasks can be submitted."""
        session_ids = [SessionManager.create_session() for _ in range(3)]
        task_ids = []
        
        try:
            # Submit multiple training tasks
            for session_id in session_ids:
                response = test_client.post('/api/train-consingan', json={
                    'session_id': session_id,
                    'iterations': 100,
                    'learning_rate': 0.0002
                })
                
                assert response.status_code == 202
                task_ids.append(response.get_json()['task_id'])
            
            # Verify all task IDs are unique
            assert len(set(task_ids)) == len(task_ids), "Task IDs should be unique"
            
            # Check status of each task
            for task_id in task_ids:
                status_response = test_client.get(f'/api/training-status/{task_id}')
                assert status_response.status_code == 200
        
        finally:
            # Cleanup
            for session_id in session_ids:
                uploads_path = SessionManager.get_session_path(session_id, "uploads")
                models_path = SessionManager.get_session_path(session_id, "models")
                if os.path.exists(uploads_path):
                    shutil.rmtree(uploads_path)
                if os.path.exists(models_path):
                    shutil.rmtree(models_path)
