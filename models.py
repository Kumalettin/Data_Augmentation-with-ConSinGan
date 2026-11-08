"""
ConSinGAN Models — orijinal makaleye birebir uyumlu mimari.

Referans: "Improved Techniques for Training Single-Image GANs"
  - Generator : ConvBlock yığını + residual bağlantı (prev_img + out)
  - Discriminator : PatchGAN — çıkış (B, 1, H', W') aktivasyon haritası
  - Loss : BCEWithLogitsLoss (patch-wise) + L1 reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temel yapı taşı
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d(k=3, p=1) → BatchNorm2d → LeakyReLU(0.2)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Generator  (residual, orijinal makale §3.1)
# ---------------------------------------------------------------------------

class ConSinGANGenerator(nn.Module):
    """
    Tek ölçek için jeneratör.

    forward(noise, prev_img):
        x      = prev_img + noise          # gürültüyü resme ekle
        out    = ConvBlocks(x)             # artık öğren
        return prev_img + out              # residual: sadece farkı ekle

    Katman 0'da prev_img sıfır tensördür.
    """

    def __init__(
        self,
        num_filters: int = 32,
        num_blocks: int = 5,
        input_channels: int = 3,
        output_channels: int = 3,
        # Geriye dönük uyumluluk için tutuldu (kullanılmıyor)
        num_scales: int = 1,
    ):
        super().__init__()
        self.num_scales      = num_scales
        self.input_channels  = input_channels
        self.output_channels = output_channels
        self.num_filters     = num_filters

        layers: list[nn.Module] = [ConvBlock(input_channels, num_filters)]
        for _ in range(num_blocks - 2):
            layers.append(ConvBlock(num_filters, num_filters))
        # Son çıkış katmanı: Conv2d + Tanh (BN yok)
        layers.append(
            nn.Sequential(
                nn.Conv2d(num_filters, output_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            )
        )
        self.body = nn.Sequential(*layers)

    def forward(
        self,
        noise: torch.Tensor,
        prev_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prev_img is None:
            prev_img = torch.zeros_like(noise)

        # prev_img boyutu noise ile eşleşmiyorsa hizala
        if prev_img.shape[-2:] != noise.shape[-2:]:
            prev_img = F.interpolate(
                prev_img, size=noise.shape[-2:], mode="bilinear", align_corners=False
            )

        # Gürültüyü conv bloklarına girmeden önce prev_img üzerine ekle (madde 3)
        x   = prev_img + noise
        out = self.body(x)

        # Residual toplam: sınırları aşmayı önlemek için clamp (madde 2)
        return torch.clamp(prev_img + out, -1.0, 1.0)

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters()) * 4


# ---------------------------------------------------------------------------
# Discriminator  (PatchGAN, orijinal makale §3.2)
# ---------------------------------------------------------------------------

class ConSinGANDiscriminator(nn.Module):
    """
    PatchGAN diskriminatör.

    Giriş : (B, 3, H, W)
    Çıkış : (B, 1, H', W')  — her piksel bir patch kararı

    Flatten / Linear / AdaptiveAvgPool YOK.
    BCEWithLogitsLoss patch haritasının tüm piksellerine uygulanır.
    """

    def __init__(
        self,
        num_filters: int = 32,
        num_blocks: int = 5,
        input_channels: int = 3,
        # Geriye dönük uyumluluk için tutuldu (kullanılmıyor)
        num_scales: int = 1,
    ):
        super().__init__()
        self.num_scales = num_scales

        layers: list[nn.Module] = [ConvBlock(input_channels, num_filters)]
        for _ in range(num_blocks - 2):
            layers.append(ConvBlock(num_filters, num_filters))
        # Son katman: Conv2d(out=1) — aktivasyon haritası
        layers.append(
            nn.Conv2d(num_filters, 1, kernel_size=3, padding=1)
        )
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)   # (B, 1, H', W')


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class ConSinGANLoss:
    """
    Patch-wise BCEWithLogitsLoss + L1 reconstruction.

    Label smoothing:
      real → 0.9,  fake → 0.1
    """

    def __init__(self, recon_weight: float = 10.0):
        self.adversarial_loss    = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.recon_weight        = recon_weight

    def generator_loss(
        self,
        fake_patches: torch.Tensor,
        original_image: torch.Tensor,
        generated_image: torch.Tensor,
    ) -> torch.Tensor:
        real_labels = torch.full_like(fake_patches, 0.9)
        adv_loss    = self.adversarial_loss(fake_patches, real_labels)
        recon_loss  = self.reconstruction_loss(generated_image, original_image)
        return adv_loss + self.recon_weight * recon_loss

    def discriminator_loss(
        self,
        real_patches: torch.Tensor,
        fake_patches: torch.Tensor,
    ) -> torch.Tensor:
        real_labels = torch.full_like(real_patches, 0.9)
        fake_labels = torch.full_like(fake_patches, 0.1)
        real_loss   = self.adversarial_loss(real_patches, real_labels)
        fake_loss   = self.adversarial_loss(fake_patches, fake_labels)
        return (real_loss + fake_loss) * 0.5
