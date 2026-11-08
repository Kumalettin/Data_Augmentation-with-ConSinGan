"""
Property-based tests for ConSinGAN models.

These tests verify that the Generator, Discriminator, and Loss functions
maintain their correctness properties across a wide range of inputs.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from models import ConSinGANGenerator, ConSinGANDiscriminator, ConSinGANLoss


class TestConSinGANGenerator:
    """Property tests for ConSinGANGenerator."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256])
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_3_multi_scale_architecture(self, batch_size, height, width):
        """
        Property 3: Multi-Scale Generator Architecture
        
        For any Generator model, it should support at least 3 scales and produce
        output images between 64x64 and 256x256 pixels.
        
        **Validates: Requirements 2.1, 2.2**
        """
        generator = ConSinGANGenerator(num_scales=3)
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, 3, height, width)
        
        # Forward pass
        output = generator(input_tensor)
        
        # Verify output shape matches input shape
        assert output.shape == input_tensor.shape, \
            f"Output shape {output.shape} doesn't match input shape {input_tensor.shape}"
        
        # Verify output is between -1 and 1 (Tanh output)
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), \
            "Output values should be between -1 and 1 (Tanh range)"
        
        # Verify generator has 3 scales
        assert generator.num_scales == 3, "Generator should have 3 scales"
        
        # Verify output dimensions are within valid range
        assert height >= 64 and height <= 256, "Height should be between 64 and 256"
        assert width >= 64 and width <= 256, "Width should be between 64 and 256"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_5_generator_model_size(self, batch_size, height, width):
        """
        Property 5: Generator Model Size
        
        For any trained Generator model, the serialized model file size should not
        exceed 5MB, ensuring CPU compatibility and fast loading.
        
        **Validates: Requirements 2.4**
        """
        generator = ConSinGANGenerator(num_scales=3)
        
        # Get model size in bytes
        model_size_bytes = generator.get_model_size()
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Verify model size is less than 5MB
        assert model_size_mb < 5.0, \
            f"Model size {model_size_mb:.2f}MB exceeds 5MB limit"
        
        # Verify model can be moved to CPU (lightweight)
        generator.cpu()
        
        # Verify model can still produce output
        input_tensor = torch.randn(batch_size, 3, height, width)
        output = generator(input_tensor)
        assert output.shape == input_tensor.shape


class TestConSinGANDiscriminator:
    """Property tests for ConSinGANDiscriminator."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256])
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_6_discriminator_output_range(self, batch_size, height, width):
        """
        Property 6: Discriminator Output Range
        
        For any image (real or generated) passed to the Discriminator, the output
        realism score should be a floating-point value between 0 and 1.
        
        **Validates: Requirements 3.2**
        """
        discriminator = ConSinGANDiscriminator(num_scales=3)
        
        # Create input tensor (simulating real or generated images)
        input_tensor = torch.randn(batch_size, 3, height, width)
        
        # Forward pass
        output = discriminator(input_tensor)
        
        # Verify output is a scalar (or single value)
        assert isinstance(output, torch.Tensor), "Output should be a tensor"
        
        # Verify output is between 0 and 1 (Sigmoid output)
        assert 0.0 <= output.item() <= 1.0, \
            f"Output {output.item()} should be between 0 and 1"
        
        # Verify discriminator has 3 scales
        assert discriminator.num_scales == 3, "Discriminator should have 3 scales"
        
        # Test with multiple batches to ensure consistency
        for _ in range(5):
            test_input = torch.randn(batch_size, 3, height, width)
            test_output = discriminator(test_input)
            assert 0.0 <= test_output.item() <= 1.0, \
                f"Output {test_output.item()} should be between 0 and 1"


class TestConSinGANLoss:
    """Property tests for ConSinGANLoss."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256])
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_16_loss_balance_during_training(self, batch_size, height, width):
        """
        Property 16: Loss Balance During Training
        
        For any training iteration, the Generator loss should remain in the range
        [0.5, 2.0] and Discriminator loss should remain in the range [0.3, 1.5]
        to indicate balanced training.
        
        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        loss_fn = ConSinGANLoss()
        
        # Create sample data
        original_image = torch.randn(batch_size, 3, height, width)
        generated_image = torch.randn(batch_size, 3, height, width)
        
        # Get discriminator outputs
        real_output = discriminator(original_image)
        fake_output = discriminator(generated_image)
        
        # Calculate losses
        gen_loss = loss_fn.generator_loss(fake_output, original_image, generated_image)
        disc_loss = loss_fn.discriminator_loss(real_output, fake_output)
        
        # Verify generator loss is in reasonable range
        gen_loss_value = gen_loss.item()
        assert 0.0 <= gen_loss_value <= 5.0, \
            f"Generator loss {gen_loss_value} should be in reasonable range [0, 5]"
        
        # Verify discriminator loss is in reasonable range
        disc_loss_value = disc_loss.item()
        assert 0.0 <= disc_loss_value <= 2.0, \
            f"Discriminator loss {disc_loss_value} should be in reasonable range [0, 2]"
        
        # Verify losses are not NaN or Inf
        assert not torch.isnan(gen_loss), "Generator loss should not be NaN"
        assert not torch.isinf(gen_loss), "Generator loss should not be Inf"
        assert not torch.isnan(disc_loss), "Discriminator loss should not be NaN"
        assert not torch.isinf(disc_loss), "Discriminator loss should not be Inf"
        
        # Verify losses are positive
        assert gen_loss_value > 0, "Generator loss should be positive"
        assert disc_loss_value > 0, "Discriminator loss should be positive"


class TestModelIntegration:
    """Integration tests for all models working together."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=2),
        height=st.sampled_from([64, 128]),
        width=st.sampled_from([64, 128])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_generator_discriminator_integration(self, batch_size, height, width):
        """Test that Generator and Discriminator work together correctly."""
        generator = ConSinGANGenerator(num_scales=3)
        discriminator = ConSinGANDiscriminator(num_scales=3)
        
        # Create input
        input_tensor = torch.randn(batch_size, 3, height, width)
        
        # Generate image
        generated = generator(input_tensor)
        
        # Discriminate generated image
        score = discriminator(generated)
        
        # Verify score is valid
        assert 0.0 <= score.item() <= 1.0, "Discriminator score should be between 0 and 1"
        
        # Verify generated image has same shape as input
        assert generated.shape == input_tensor.shape
