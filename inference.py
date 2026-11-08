"""
Inference module for ConSinGAN — çok ölçekli zincir üretim.

Yeni mimari her ölçek için ayrı bir Generator kullanır.
Üretim: G_0(noise_0) → upsample → G_1(noise_1, prev) → ... → G_N(noise_N, prev)
z_opt (sabit gürültü) varsa reconstruction path, yoksa rastgele gürültü kullanılır.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import base64
import logging
from io import BytesIO
from PIL import Image
from typing import List

from models import ConSinGANGenerator, ConSinGANDiscriminator
from session_manager import SessionManager
from device_manager import DeviceManager

logger = logging.getLogger(__name__)


class ConSinGANInference:
    """
    Çok ölçekli ConSinGAN inference.

    load_model() → tüm ölçeklerin G'lerini yükler
    generate()   → zincir forward pass ile görsel üretir
    """

    def __init__(self, session_id: str, device: str = None):
        if not session_id:
            raise ValueError("session_id cannot be empty")

        self.session_id  = session_id
        self.device      = device or DeviceManager.get_optimal_device()

        # Çok ölçekli state
        self.generators:  List[ConSinGANGenerator]     = []
        self.noise_amps:  List[float]                  = []
        self.scales:      List[tuple]                  = []
        self.z_opts:      List[torch.Tensor]           = []   # varsa sabit gürültü
        self.config:      dict                         = {}

        # Geriye dönük uyumluluk (tek G)
        self.generator    = None
        self.discriminator = None

        logger.info(
            f"Initialized ConSinGANInference for session {session_id} "
            f"on device {self.device}"
        )

    # ------------------------------------------------------------------
    # Model yükleme
    # ------------------------------------------------------------------

    def load_model(self, session_id: str = None) -> bool:
        """
        Tüm ölçeklerin generator ağırlıklarını diskten yükler.
        generator_0.pth, generator_1.pth, ... varsa çok ölçekli mod.
        Sadece generator.pth varsa tek ölçekli (eski) mod.
        """
        if session_id is None:
            session_id = self.session_id

        models_path  = SessionManager.get_session_path(session_id, "models")
        config_path  = os.path.join(models_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        num_scales = self.config.get('num_scales', 1)
        self.scales = [tuple(s) for s in self.config.get('scales', [])]
        self.noise_amps = self.config.get('noise_amps', [1.0] * num_scales)

        logger.info(f"Loading {num_scales} scale(s) for session {session_id}")

        # ── Çok ölçekli mod: generator_0.pth, generator_1.pth, ... ───────
        multi_path_0 = os.path.join(models_path, "generator_0.pth")
        if os.path.exists(multi_path_0):
            self.generators = []
            for idx in range(num_scales):
                g_path = os.path.join(models_path, f"generator_{idx}.pth")
                if not os.path.exists(g_path):
                    raise FileNotFoundError(f"Generator {idx} not found: {g_path}")

                g = ConSinGANGenerator(num_filters=32, num_blocks=5).to(self.device)
                state = torch.load(g_path, map_location=self.device, weights_only=False)
                g.load_state_dict(state)
                g.eval()
                self.generators.append(g)
                logger.info(f"  Loaded generator_{idx}.pth")

            # z_opts varsa yükle
            z_path = os.path.join(models_path, "z_opts.pth")
            if os.path.exists(z_path):
                z_list = torch.load(z_path, map_location=self.device, weights_only=False)
                self.z_opts = [z.to(self.device) for z in z_list]
                logger.info(f"  Loaded z_opts ({len(self.z_opts)} tensors)")

            # Geriye dönük uyumluluk için son G'yi de ata
            self.generator = self.generators[-1]

        # ── Tek ölçekli mod (eski kayıt): sadece generator.pth ───────────
        else:
            g_path = os.path.join(models_path, "generator.pth")
            if not os.path.exists(g_path):
                raise FileNotFoundError(f"Generator not found: {g_path}")

            g = ConSinGANGenerator(num_filters=32, num_blocks=5).to(self.device)
            state = torch.load(g_path, map_location=self.device, weights_only=False)
            g.load_state_dict(state)
            g.eval()
            self.generators = [g]
            self.generator   = g
            logger.info("  Loaded single generator.pth (legacy mode)")

        logger.info(f"Models loaded successfully for session {session_id}")
        return True

    # ------------------------------------------------------------------
    # Üretim
    # ------------------------------------------------------------------

    def generate(self, num_images: int = 5) -> list:
        """
        Çok ölçekli zincir forward pass ile görsel üretir.

        Her üretimde:
          - z_opt varsa: z_opt zinciri (reconstruction path) + küçük rastgele pertürbasyon
          - z_opt yoksa: tamamen rastgele gürültü zinciri

        Döner: [(H,W,3) uint8 NumPy] listesi
        """
        if not self.generators:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if num_images < 1:
            raise ValueError("num_images must be at least 1")

        num_scales  = len(self.generators)
        use_z_opts  = len(self.z_opts) == num_scales
        generated_images = []

        with torch.no_grad():
            for img_idx in range(num_images):
                prev = None

                for idx, g in enumerate(self.generators):
                    amp = self.noise_amps[idx] if idx < len(self.noise_amps) else 1.0

                    # Gürültü boyutunu belirle
                    if self.scales and idx < len(self.scales):
                        h_i, w_i = self.scales[idx]
                    elif use_z_opts:
                        h_i, w_i = self.z_opts[idx].shape[-2:]
                    else:
                        # Fallback: son ölçek 256×256
                        h_i = w_i = 256

                    if use_z_opts:
                        # z_opt + küçük rastgele pertürbasyon (çeşitlilik için)
                        perturb = 0.1 * amp * torch.randn(
                            1, 3, h_i, w_i, device=self.device
                        )
                        noise_i = self.z_opts[idx] + perturb
                    else:
                        noise_i = amp * torch.randn(1, 3, h_i, w_i, device=self.device)

                    prev = g(noise_i, prev)

                    # Bir sonraki ölçeğe upsample
                    if idx < num_scales - 1:
                        if self.scales and (idx + 1) < len(self.scales):
                            next_h, next_w = self.scales[idx + 1]
                        else:
                            next_h = next_w = prev.shape[-1] * 2
                        prev = F.interpolate(
                            prev, size=(next_h, next_w),
                            mode="bilinear", align_corners=False
                        )

                # prev: (1, 3, H, W) float [-1, 1]
                arr = prev.squeeze(0).cpu().numpy()
                arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
                arr = np.transpose(arr, (1, 2, 0))   # CHW → HWC
                generated_images.append(arr)
                logger.debug(f"Generated image {img_idx+1}/{num_images}")

        logger.info(f"Successfully generated {num_images} images")
        return generated_images

    # ------------------------------------------------------------------
    # Base64 encode
    # ------------------------------------------------------------------

    def generate_base64(self, num_images: int = 5) -> list:
        """Görsel üretip base64 listesi döner."""
        try:
            images = self.generate(num_images)
            result = []
            for img_np in images:
                pil_image = Image.fromarray(img_np, mode='RGB')
                buffer    = BytesIO()
                pil_image.save(buffer, format='JPEG', quality=95)
                buffer.seek(0)
                b64      = base64.b64encode(buffer.getvalue()).decode('utf-8')
                result.append({
                    'image': f"data:image/jpeg;base64,{b64}",
                    'label': 'Sentetik (AI)',
                    'ssim':  0.0,
                })
            logger.info(
                f"Successfully generated and encoded {num_images} images as base64"
            )
            return result
        except Exception as e:
            logger.error(f"Base64 encoding failed: {e}")
            raise RuntimeError(f"Failed to generate base64 images: {e}")
