"""
Training Module for ConSinGAN — Z_opt + Concurrent Training düzeltmesi.

Kritik kurallar (orijinal makale §3):
  1. Z_opt (sabit gürültü) her ölçek için ayrı tutulur.
     - Adversarial path : Generator(rand_noise,  prev_fake)   → fake_adv
     - Reconstruction path: Generator(z_opt,      prev_recon)  → fake_recon
     L1 loss SADECE fake_recon üzerinden hesaplanır.

  2. noise_amp hesabı: RMSE(real_scaled, prev_recon_upsampled)
     Rastgele fake değil, z_opt zincirinden gelen prev_recon kullanılır.

  3. Gradient akışı:
     - D adımı : fake_adv.detach()  (D'nin gradientleri G'ye akmasın)
     - G adımı : detach YOK        (gradientler tüm alt katmanlara akar)
"""

import os
import json
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from threading import Thread
from queue import Queue
import uuid

from models import ConSinGANGenerator, ConSinGANDiscriminator, ConSinGANLoss
from texture_preservation import TexturePreservationOptimizer
from device_manager import DeviceManager
from session_manager import SessionManager
from model_manager import ModelManager

logger = logging.getLogger(__name__)

model_manager = ModelManager(base_path="model_library")

# ---------------------------------------------------------------------------
# Dinamik ölçek listesi
# ---------------------------------------------------------------------------

SCALE_FACTOR = 0.75
MIN_SIZE     = 25


def compute_scales(real_h: int, real_w: int) -> List[tuple]:
    """
    Görüntü boyutuna göre küçükten büyüğe çözünürlük piramidi üretir.
    num_scales = ceil( log(MIN_SIZE / min(H,W)) / log(SCALE_FACTOR) )
    """
    min_dim    = min(real_h, real_w)
    num_scales = math.ceil(math.log(MIN_SIZE / min_dim, SCALE_FACTOR))
    num_scales = max(num_scales, 1)

    scales = []
    for i in range(num_scales + 1):
        r  = SCALE_FACTOR ** (num_scales - i)
        sh = max(MIN_SIZE, int(round(real_h * r)))
        sw = max(MIN_SIZE, int(round(real_w * r)))
        scales.append((sh, sw))

    return scales   # [(h0,w0), ..., (H,W)]


# ---------------------------------------------------------------------------
# ConSinGANTrainer
# ---------------------------------------------------------------------------

class ConSinGANTrainer:
    """
    Orijinal ConSinGAN eğitim döngüsü.

    Her ölçek için ayrı G ve D eğitilir.
    Z_opt (sabit gürültü) reconstruction path'ini adversarial path'ten ayırır.
    """

    def __init__(self, device: str):
        self.device = device
        self.loss_fn = ConSinGANLoss(recon_weight=10.0)

        self.generators:     List[ConSinGANGenerator]     = []
        self.discriminators: List[ConSinGANDiscriminator] = []
        self.noise_amps:     List[float]                  = []
        self.z_opts:         List[torch.Tensor]           = []   # sabit gürültüler
        self.scales:         List[tuple]                  = []

        self.training_state: Dict[str, Any] = {
            'current_iteration': 0,
            'total_iterations':  0,
            'loss_g':            0.0,
            'loss_d':            0.0,
            'ssim_score':        0.0,
        }

        logger.info(f"ConSinGANTrainer initialized on device: {device}")

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    def _img_to_tensor(self, img_np: np.ndarray, h: int, w: int) -> torch.Tensor:
        """NumPy (H,W,3) uint8 → (1,3,h,w) float32 tensor [-1,1]."""
        import cv2
        resized = cv2.resize(img_np, (w, h))
        arr = resized.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _tensor_to_np(self, t: torch.Tensor) -> np.ndarray:
        """(1,3,H,W) float32 tensor [-1,1] → (H,W,3) uint8 NumPy."""
        arr = t.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Forward pass yardımcıları
    # ------------------------------------------------------------------

    def _forward_chain_adv(
        self,
        rand_noise_top: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        """
        Adversarial path: her ölçekte rastgele gürültü kullanır.
        Gradient akışı AÇIK — detach yok.
        """
        prev = None
        for idx in range(scale_idx + 1):
            h_i, w_i = self.scales[idx]
            amp_i    = self.noise_amps[idx]

            if idx == scale_idx:
                noise_i = rand_noise_top
            else:
                noise_i = amp_i * torch.randn(
                    rand_noise_top.shape[0], rand_noise_top.shape[1],
                    h_i, w_i, device=self.device
                )

            prev = self.generators[idx](noise_i, prev)

            if idx < scale_idx:
                next_h, next_w = self.scales[idx + 1]
                prev = F.interpolate(
                    prev, size=(next_h, next_w),
                    mode="bilinear", align_corners=False
                )

        return prev   # fake_adv — gradient akışı açık

    def _forward_chain_recon(
        self,
        scale_idx: int,
    ) -> torch.Tensor:
        """
        Reconstruction path: her ölçekte z_opt kullanır.
        Gradient akışı AÇIK — detach yok.
        """
        prev = None
        for idx in range(scale_idx + 1):
            z_i = self.z_opts[idx]

            prev = self.generators[idx](z_i, prev)

            if idx < scale_idx:
                next_h, next_w = self.scales[idx + 1]
                prev = F.interpolate(
                    prev, size=(next_h, next_w),
                    mode="bilinear", align_corners=False
                )

        return prev   # fake_recon — gradient akışı açık

    def _recon_no_grad(self, scale_idx: int) -> torch.Tensor:
        """
        Gradient hesabı olmadan reconstruction zinciri.
        noise_amp hesabı ve preview üretimi için kullanılır.
        """
        with torch.no_grad():
            prev = None
            for idx in range(scale_idx + 1):
                z_i = self.z_opts[idx]
                prev = self.generators[idx](z_i, prev)
                if idx < scale_idx:
                    next_h, next_w = self.scales[idx + 1]
                    prev = F.interpolate(
                        prev, size=(next_h, next_w),
                        mode="bilinear", align_corners=False
                    )
        return prev

    # ------------------------------------------------------------------
    # Ana eğitim fonksiyonu
    # ------------------------------------------------------------------

    def train(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        session_id: str,
        progress_callback: Optional[Callable] = None,
        model_name: str = None,
    ) -> Dict[str, Any]:
        try:
            import cv2

            iterations      = config.get('iterations', 2000)
            learning_rate   = config.get('learning_rate', 0.0005)
            iters_per_scale = config.get('iters_per_scale', None)

            if not (100 <= iterations <= 50000):
                iterations = 2000
            if not (1e-5 <= learning_rate <= 0.01):
                learning_rate = 0.0005

            real_h, real_w = image.shape[:2]

            # ── 1. Ölçek piramidi ─────────────────────────────────────────
            self.scales    = compute_scales(real_h, real_w)
            num_scales     = len(self.scales)
            logger.info(f"Scales ({num_scales}): {self.scales}")

            if iters_per_scale:
                s_iters = [iters_per_scale] * num_scales
            else:
                base      = iterations // num_scales
                remainder = iterations % num_scales
                s_iters   = [max(10, base)] * (num_scales - 1) + [max(10, base + remainder)]

            total_iters = sum(s_iters)
            self.training_state['total_iterations'] = total_iters

            session_models_path = SessionManager.get_session_path(session_id, "models")
            Path(session_models_path).mkdir(parents=True, exist_ok=True)

            # ── 2. G ve D listelerini oluştur ─────────────────────────────
            self.generators     = []
            self.discriminators = []
            self.noise_amps     = []
            self.z_opts         = []

            for _ in range(num_scales):
                g = ConSinGANGenerator(num_filters=32, num_blocks=5).to(self.device)
                d = ConSinGANDiscriminator(num_filters=32, num_blocks=5).to(self.device)
                self.generators.append(g)
                self.discriminators.append(d)

            # ── 3. Ölçek döngüsü ──────────────────────────────────────────
            total_done = 0

            for scale_idx in range(num_scales):
                sh, sw = self.scales[scale_idx]
                logger.info(
                    f"── Scale {scale_idx+1}/{num_scales}  {sw}×{sh}: "
                    f"{s_iters[scale_idx]} iters ──"
                )

                real_tensor = self._img_to_tensor(image, sh, sw)
                B, C, H, W  = real_tensor.shape

                # ── Z_opt: bu ölçek için sabit gürültü ────────────────────
                z_opt = torch.randn(B, C, H, W, device=self.device)
                self.z_opts.append(z_opt)

                # ── Noise amplitude ────────────────────────────────────────
                # Kural 2: RMSE(real, prev_recon_upsampled) — z_opt zinciri
                if scale_idx == 0:
                    noise_amp = 1.0
                else:
                    # z_opt zinciriyle önceki ölçeğe kadar üret, bu ölçeğe upsample et
                    prev_recon = self._recon_no_grad(scale_idx - 1)
                    upsampled  = F.interpolate(
                        prev_recon, size=(H, W),
                        mode="bilinear", align_corners=False
                    )
                    rmse      = torch.sqrt(torch.mean((real_tensor - upsampled) ** 2))
                    noise_amp = rmse.item()

                self.noise_amps.append(noise_amp)
                logger.info(f"  noise_amp = {noise_amp:.4f}")

                # z_opt'u noise_amp ile ölçekle
                z_opt_scaled = noise_amp * z_opt

                # ── Optimizer listesi (concurrent training) ────────────────
                opt_g_cur = optim.Adam(
                    self.generators[scale_idx].parameters(),
                    lr=learning_rate, betas=(0.5, 0.999)
                )
                opt_d_cur = optim.Adam(
                    self.discriminators[scale_idx].parameters(),
                    lr=learning_rate, betas=(0.5, 0.999)
                )

                # Önceki katmanların G optimize edicileri — LR kademeli düşer
                prev_g_opts = []
                for prev_idx in range(scale_idx):
                    lr_scaled = learning_rate * (0.1 ** (scale_idx - prev_idx))
                    prev_g_opts.append(
                        optim.Adam(
                            self.generators[prev_idx].parameters(),
                            lr=lr_scaled, betas=(0.5, 0.999)
                        )
                    )

                log_interval = max(1, s_iters[scale_idx] // 10)

                for it in range(s_iters[scale_idx]):
                    # Adversarial için rastgele gürültü (her iterasyonda yeni)
                    rand_noise = noise_amp * torch.randn(B, C, H, W, device=self.device)

                    try:
                        # ── Adversarial path (rand_noise) ──────────────────
                        # Kural 3: D adımı için detach, G adımı için detach yok
                        fake_adv = self._forward_chain_adv(rand_noise, scale_idx)

                        # ── Discriminator adımı ────────────────────────────
                        real_p = self.discriminators[scale_idx](real_tensor)
                        fake_p = self.discriminators[scale_idx](fake_adv.detach())  # D için detach
                        loss_d = self.loss_fn.discriminator_loss(real_p, fake_p)

                        opt_d_cur.zero_grad()
                        loss_d.backward()
                        opt_d_cur.step()

                        # ── Reconstruction path (z_opt) ────────────────────
                        # Kural 1: L1 loss SADECE recon path üzerinden
                        fake_adv_g  = self._forward_chain_adv(rand_noise, scale_idx)
                        fake_recon  = self._forward_chain_recon(scale_idx)

                        fake_p_g = self.discriminators[scale_idx](fake_adv_g)  # detach yok
                        loss_g   = self.loss_fn.generator_loss(
                            fake_p_g, real_tensor, fake_recon   # recon path ile L1
                        )

                        # Tüm G optimize edicilerini sıfırla
                        opt_g_cur.zero_grad()
                        for opt in prev_g_opts:
                            opt.zero_grad()

                        loss_g.backward()   # gradient tüm alt katmanlara akar

                        opt_g_cur.step()
                        for opt in prev_g_opts:
                            opt.step()

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning("GPU OOM — iterasyon atlandı")
                            if self.device == "cuda":
                                torch.cuda.empty_cache()
                            continue
                        raise

                    total_done += 1
                    self.training_state['current_iteration'] = total_done
                    self.training_state['loss_g'] = loss_g.item()
                    self.training_state['loss_d'] = loss_d.item()

                    if (it + 1) % log_interval == 0:
                        progress = int(total_done / total_iters * 100)
                        logger.info(
                            f"  [{it+1}/{s_iters[scale_idx]}] ({progress}%) "
                            f"G={loss_g.item():.4f}  D={loss_d.item():.4f}"
                        )
                        if progress_callback:
                            progress_callback({
                                'progress':         progress,
                                'loss_g':           loss_g.item(),
                                'loss_d':           loss_d.item(),
                                'iteration':        total_done,
                                'total_iterations': total_iters,
                            })

                # ── Scale bitti: ara görsel kaydet (z_opt zinciri) ─────────
                preview_tensor = self._recon_no_grad(scale_idx)
                preview_np     = self._tensor_to_np(preview_tensor)
                preview_bgr    = cv2.cvtColor(preview_np, cv2.COLOR_RGB2BGR)
                scale_num      = scale_idx + 1
                preview_path   = os.path.join(session_models_path, f"{scale_num}_scale.jpg")
                cv2.imwrite(preview_path, preview_bgr)
                logger.info(f"  Preview saved: {preview_path}")

                if scale_idx == num_scales - 1:
                    cv2.imwrite(
                        os.path.join(session_models_path, "final_image.jpg"),
                        preview_bgr
                    )

                if progress_callback:
                    progress_callback({
                        'progress':         int(total_done / total_iters * 100),
                        'loss_g':           loss_g.item(),
                        'loss_d':           loss_d.item(),
                        'iteration':        total_done,
                        'total_iterations': total_iters,
                        'scale_preview':    preview_path,
                        'scale_num':        scale_num,
                    })

            # ── 4. SSIM ───────────────────────────────────────────────────
            final_tensor = self._recon_no_grad(num_scales - 1)
            final_np     = self._tensor_to_np(final_tensor)
            image_256    = cv2.resize(image, (256, 256))
            final_256    = cv2.resize(final_np, (256, 256))
            ssim_score   = self.calculate_ssim(image_256, final_256)
            self.training_state['ssim_score'] = ssim_score

            self.save_model(session_id)

            logger.info(
                f"Eğitim tamamlandı. SSIM={ssim_score:.4f}  "
                f"G={loss_g.item():.4f}  D={loss_d.item():.4f}"
            )
            return {
                'success':    True,
                'ssim_score': ssim_score,
                'loss_g':     loss_g.item(),
                'loss_d':     loss_d.item(),
                'message':    f'Training completed. SSIM: {ssim_score:.4f}',
            }

        except Exception as e:
            logger.error(f"Training failed for session {session_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error':   str(e),
                'message': f'Training failed: {str(e)}',
            }

    # ------------------------------------------------------------------
    # Model kaydetme
    # ------------------------------------------------------------------

    def save_model(self, session_id: str, model_name: str = None) -> bool:
        try:
            models_path = SessionManager.get_session_path(session_id, "models")
            Path(models_path).mkdir(parents=True, exist_ok=True)

            for idx, (g, d) in enumerate(zip(self.generators, self.discriminators)):
                torch.save(g.state_dict(),
                           os.path.join(models_path, f"generator_{idx}.pth"))
                torch.save(d.state_dict(),
                           os.path.join(models_path, f"discriminator_{idx}.pth"))

            # Geriye dönük uyumluluk
            if self.generators:
                torch.save(self.generators[-1].state_dict(),
                           os.path.join(models_path, "generator.pth"))
                torch.save(self.discriminators[-1].state_dict(),
                           os.path.join(models_path, "discriminator.pth"))

            # z_opts'u kaydet (inference için gerekli)
            z_opts_cpu = [z.cpu() for z in self.z_opts]
            torch.save(z_opts_cpu, os.path.join(models_path, "z_opts.pth"))

            config = {
                'num_scales':      len(self.scales),
                'scales':          self.scales,
                'noise_amps':      self.noise_amps,
                'input_channels':  3,
                'output_channels': 3,
                'device':          self.device,
                'timestamp':       datetime.now().isoformat(),
                'training_state':  self.training_state,
            }
            with open(os.path.join(models_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)

            if model_name:
                try:
                    ref_image_path = os.path.join(
                        SessionManager.get_session_path(session_id, "uploads"),
                        "original.jpg"
                    )
                    lib_model_id = model_manager.save_trained_model(
                        model_name=model_name,
                        generator_state=self.generators[-1].state_dict(),
                        discriminator_state=self.discriminators[-1].state_dict(),
                        reference_image_path=ref_image_path,
                        config=config,
                        description=f"Trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    )

                    # Tüm ölçeklerin G/D ağırlıklarını ve z_opts'u kütüphane klasörüne kopyala
                    if lib_model_id:
                        lib_path = os.path.join("model_library", lib_model_id)
                        for idx, (g, d) in enumerate(zip(self.generators, self.discriminators)):
                            torch.save(g.state_dict(),
                                       os.path.join(lib_path, f"generator_{idx}.pth"))
                            torch.save(d.state_dict(),
                                       os.path.join(lib_path, f"discriminator_{idx}.pth"))
                        z_opts_cpu = [z.cpu() for z in self.z_opts]
                        torch.save(z_opts_cpu, os.path.join(lib_path, "z_opts.pth"))
                        logger.info(f"All scale weights copied to library: {lib_path}")

                except Exception as e:
                    logger.warning(f"Model kütüphanesine kaydedilemedi: {e}")

            logger.info(f"Models saved for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save models for session {session_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # SSIM
    # ------------------------------------------------------------------

    def calculate_ssim(self, original: np.ndarray, generated: np.ndarray) -> float:
        try:
            if original.dtype != np.uint8:
                original  = np.clip(original,  0, 255).astype(np.uint8)
            if generated.dtype != np.uint8:
                generated = np.clip(generated, 0, 255).astype(np.uint8)
            if original.shape != generated.shape:
                import cv2
                generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
            texture_optimizer = TexturePreservationOptimizer(original)
            return float(texture_optimizer.get_ssim_score(generated))
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0.0

    def get_training_state(self) -> Dict[str, Any]:
        return self.training_state.copy()


# ---------------------------------------------------------------------------
# TrainingTaskManager
# ---------------------------------------------------------------------------

class TrainingTaskManager:
    """Arka planda eğitim görevlerini yöneten thread havuzu."""

    def __init__(self, max_workers: int = 2):
        self.task_queue   = Queue()
        self.active_tasks: Dict[str, Any] = {}
        self.max_workers  = max_workers
        self.workers:     List[Thread] = []
        self._running     = True

        for i in range(max_workers):
            w = Thread(target=self._worker_loop, daemon=True,
                       name=f"TrainingWorker-{i}")
            w.start()
            self.workers.append(w)

        logger.info(f"TrainingTaskManager initialized with {max_workers} workers")

    def submit_training_task(self, session_id: str, config: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        task = {
            'task_id':        task_id,
            'session_id':     session_id,
            'config':         config,
            'status':         'queued',
            'progress':       0,
            'loss_g':         0.0,
            'loss_d':         0.0,
            'eta_seconds':    0,
            'error_message':  None,
            'start_time':     datetime.now(),
            'end_time':       None,
            'scale_previews': [],
        }
        self.active_tasks[task_id] = task
        self.task_queue.put(task)
        logger.info(f"Training task submitted: {task_id} for session {session_id}")
        return task_id

    def _worker_loop(self):
        while self._running:
            try:
                task = self.task_queue.get(timeout=1)
                try:
                    self._execute_training(task)
                except Exception as e:
                    task['status']        = 'failed'
                    task['error_message'] = str(e)
                    logger.error(f"Training task {task['task_id']} failed: {e}")
                finally:
                    task['end_time'] = datetime.now()
                    self.task_queue.task_done()
            except Exception:
                continue

    def _execute_training(self, task: Dict[str, Any]):
        task['status']     = 'training'
        task['start_time'] = datetime.now()

        try:
            device     = DeviceManager.get_optimal_device()
            session_id = task['session_id']
            image_path = os.path.join(
                SessionManager.get_session_path(session_id, "uploads"), "original.jpg"
            )

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Original image not found: {image_path}")

            import cv2
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            trainer = ConSinGANTrainer(device)

            def progress_callback(data: Dict[str, Any]):
                task['progress'] = data.get('progress', 0)
                task['loss_g']   = data.get('loss_g', 0.0)
                task['loss_d']   = data.get('loss_d', 0.0)

                preview_path = data.get('scale_preview')
                if preview_path and os.path.exists(preview_path):
                    import cv2 as _cv2, base64 as _b64
                    img = _cv2.imread(preview_path)
                    if img is not None:
                        _, buf = _cv2.imencode(
                            '.jpg', img, [int(_cv2.IMWRITE_JPEG_QUALITY), 80]
                        )
                        b64       = _b64.b64encode(buf).decode('utf-8')
                        scale_num = data.get('scale_num', len(task['scale_previews']) + 1)
                        existing  = next(
                            (p for p in task['scale_previews']
                             if p['scale_num'] == scale_num), None
                        )
                        entry = {
                            'scale_num': scale_num,
                            'image':     f"data:image/jpeg;base64,{b64}",
                        }
                        if existing:
                            existing.update(entry)
                        else:
                            task['scale_previews'].append(entry)

                if task['progress'] > 0:
                    elapsed    = (datetime.now() - task['start_time']).total_seconds()
                    total_time = elapsed / (task['progress'] / 100.0)
                    task['eta_seconds'] = int(total_time - elapsed)

            result = trainer.train(
                image=image,
                config=task['config'],
                session_id=session_id,
                progress_callback=progress_callback,
            )

            if result.get('success', False):
                model_name = task['config'].get('model_name')
                if model_name:
                    trainer.save_model(session_id, model_name=model_name)
                    task['model_id'] = (
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
                    )
                else:
                    trainer.save_model(session_id)

                task['status']   = 'completed'
                task['progress'] = 100
                logger.info(f"Training task {task['task_id']} completed")
            else:
                task['status']        = 'failed'
                task['error_message'] = result.get('error', 'Unknown error')

        except Exception as e:
            task['status']        = 'failed'
            task['error_message'] = str(e)
            logger.error(f"Training task {task['task_id']} execution failed: {e}")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.active_tasks:
            return {
                'task_id':       task_id,
                'status':        'not_found',
                'error_message': f'Task {task_id} not found',
            }
        task = self.active_tasks[task_id]
        return {
            'task_id':        task['task_id'],
            'session_id':     task['session_id'],
            'status':         task['status'],
            'progress':       task['progress'],
            'loss_g':         task['loss_g'],
            'loss_d':         task['loss_d'],
            'eta_seconds':    task['eta_seconds'],
            'error_message':  task['error_message'],
            'scale_previews': task.get('scale_previews', []),
            'start_time':     task['start_time'].isoformat() if task['start_time'] else None,
            'end_time':       task['end_time'].isoformat()   if task['end_time']   else None,
        }

    def _get_device(self) -> str:
        return DeviceManager.get_optimal_device()

    def shutdown(self):
        logger.info("Shutting down TrainingTaskManager")
        self._running = False
        self.task_queue.join()
        for w in self.workers:
            w.join(timeout=5)
        logger.info("TrainingTaskManager shutdown complete")
