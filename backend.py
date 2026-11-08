import os
import io
import base64
import random
import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from training import TrainingTaskManager, ConSinGANTrainer
from models import ConSinGANGenerator, ConSinGANDiscriminator
from inference import ConSinGANInference
from model_manager import ModelManager
from device_manager import DeviceManager
from session_manager import SessionManager

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize model manager
model_manager = ModelManager(base_path="model_library")

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
# Tüm originlere izin ver (Frontend ve Backend farklı portlarda ise gerekli)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize task manager for background training
task_manager = TrainingTaskManager(max_workers=2)

# --- PyTorch and Device Initialization ---
try:
    import torch
    import torchvision
    
    # Detect device
    device_manager = DeviceManager()
    device = device_manager.get_optimal_device()
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Torchvision version: {torchvision.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Optimal device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
except ImportError as e:
    logger.error(f"Failed to import PyTorch or torchvision: {e}")
    raise
except Exception as e:
    logger.error(f"Error during device initialization: {e}")
    raise

# --- Helper Functions ---

def base64_to_cv2(base64_string):
    """Base64 string'i OpenCV Görüntüsüne (numpy array) çevirir"""
    try:
        # data:image/jpeg;base64, kısmını at
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def cv2_to_base64(img):
    """OpenCV Görüntüsünü Base64 string'e çevirir"""
    try:
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def apply_augmentations(img_cv, techniques):
    """Görüntüye istenen teknikleri uygular"""
    
    # OpenCV'den PIL formatına çevir (Parlaklık ve Kontrast işlemleri için PIL daha pratik)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    applied = []
    
    # İlgili teknikleri karıştır ve sırayla (ya da rastgele bir alt kümesine) uygula
    random.shuffle(techniques)
    
    # %70 ihtimalle her tekniği uygulama kararı (daha fazla çeşitlilik için)
    active_techniques = [t for t in techniques if random.random() > 0.3]
    
    # Eğer filtrelemeden dolayı hiç teknik kalmadıysa en az birini seç
    if not active_techniques and techniques:
        active_techniques = [random.choice(techniques)]

    for tech in active_techniques:
        if tech == 'rotate':
            angle = random.uniform(-45, 45)
            # OpenCV ile Döndürme
            h, w = img_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_cv = cv2.warpAffine(img_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            applied.append(f'🔄 Döndürme ({angle:.1f}°)')
            
        elif tech == 'flip':
            mode = random.choice([0, 1]) # 0: Dikey, 1: Yatay
            img_cv = cv2.flip(img_cv, mode)
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
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
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            applied.append(f'🌫️ Bulanıklık ({kernel_size}x{kernel_size})')
            
        elif tech == 'noise':
            # Gaussian Noise Ekleme
            row, col, ch = img_cv.shape
            mean = 0
            var = random.uniform(10, 50)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch).astype(np.float32)
            
            noisy = img_cv.astype(np.float32) + gauss
            # Değerleri 0-255 arasına kırp
            np.clip(noisy, 0, 255, out=noisy)
            img_cv = noisy.astype(np.uint8)
            
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            applied.append(f'📡 Gürültü')

    return img_cv, applied

# --- API Routes (MUST BE BEFORE STATIC ROUTES) ---

@app.route('/api/augment', methods=['POST'])
def augment():
    data = request.json
    
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "Geçersiz İstek: Görsel bulunamadı"}), 400
        
    base64_image = data['image']
    techniques = data.get('techniques', [])
    num_images = data.get('num_images', 5)
    
    if not techniques:
        return jsonify({"success": False, "error": "En az bir augmentation tekniği seçmelisiniz"}), 400
        
    if num_images > 50:
        num_images = 50
        
    original_cv2_img = base64_to_cv2(base64_image)
    if original_cv2_img is None:
        return jsonify({"success": False, "error": "Görsel işlenemedi / Geçersiz format"}), 400

    results = []
    
    for i in range(num_images):
        img_copy = original_cv2_img.copy()
        aug_img, applied_techs = apply_augmentations(img_copy, techniques)
        res_base64 = cv2_to_base64(aug_img)
        
        if res_base64:
            results.append({
                "image": res_base64,
                "techniques": applied_techs,
                "filename": f"augmented_python_{i+1}.jpg"
            })
            
    return jsonify({
        "success": True,
        "images": results,
        "count": len(results)
    })


def validate_training_config(config):
    """Validate training configuration parameters."""
    errors = []
    
    iterations = config.get('iterations', 1000)
    learning_rate = config.get('learning_rate', 0.0002)
    batch_size = config.get('batch_size', 1)
    num_scales = config.get('num_scales', 3)
    
    if not isinstance(iterations, int) or iterations < 100 or iterations > 10000:
        errors.append("Iterations must be between 100 and 10000")
    
    if not isinstance(learning_rate, (int, float)) or learning_rate < 0.00001 or learning_rate > 0.001:
        errors.append("Learning rate must be between 0.00001 and 0.001")
    
    if not isinstance(batch_size, int) or batch_size < 1:
        errors.append("Batch size must be at least 1")
    
    if not isinstance(num_scales, int) or num_scales < 1 or num_scales > 10:
        errors.append("Number of scales must be between 1 and 5")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, None


@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new session for image processing."""
    try:
        session_id = SessionManager.create_session()
        logger.info(f"New session created: {session_id}")
        
        return jsonify({
            "success": True,
            "session_id": session_id
        }), 201
    
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload an image for a session."""
    try:
        data = request.json
        
        if not data or 'session_id' not in data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing session_id or image"
            }), 400
        
        session_id = data['session_id']
        base64_image = data['image']
        
        if not SessionManager.session_exists(session_id):
            return jsonify({
                "success": False,
                "error": f"Session {session_id} does not exist"
            }), 404
        
        img_cv = base64_to_cv2(base64_image)
        if img_cv is None:
            return jsonify({
                "success": False,
                "error": "Invalid image format"
            }), 400
        
        uploads_path = SessionManager.get_session_path(session_id, 'uploads')
        image_path = os.path.join(uploads_path, 'original.jpg')
        
        cv2.imwrite(image_path, img_cv)
        
        logger.info(f"Image uploaded for session {session_id}")
        
        return jsonify({
            "success": True,
            "message": "Image uploaded successfully",
            "image_path": image_path
        }), 200
    
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/train-consingan', methods=['POST'])
def train_consingan():
    """Start asynchronous training of ConSinGAN model."""
    try:
        data = request.json
        
        if not data or 'session_id' not in data:
            return jsonify({
                "success": False,
                "error": "Missing session_id"
            }), 400
        
        session_id = data['session_id']
        
        if not SessionManager.session_exists(session_id):
            return jsonify({
                "success": False,
                "error": f"Session {session_id} does not exist"
            }), 404
        
        config = {
            'iterations': data.get('iterations', 1000),
            'learning_rate': data.get('learning_rate', 0.0002),
            'batch_size': data.get('batch_size', 1),
            'num_scales': data.get('num_scales', 7),
            'model_name': data.get('model_name', None)  # Model adı (opsiyonel)
        }
        
        is_valid, error_msg = validate_training_config(config)
        if not is_valid:
            logger.warning(f"Invalid training config for session {session_id}: {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400
        
        task_id = task_manager.submit_training_task(session_id, config)
        
        logger.info(f"Training task {task_id} submitted for session {session_id}")
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "message": "Training started in background"
        }), 202
    
    except Exception as e:
        logger.error(f"Error in train_consingan: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/training-status/<task_id>', methods=['GET'])
def training_status(task_id):
    """Get the status of a training task."""
    try:
        status = task_manager.get_task_status(task_id)
        
        if status.get('status') == 'not_found':
            logger.warning(f"Task {task_id} not found")
            return jsonify({
                "success": False,
                "error": f"Task {task_id} not found"
            }), 404
        
        return jsonify({
            "success": True,
            "status": status['status'],
            "progress": status.get('progress', 0),
            "loss_g": status.get('loss_g', 0.0),
            "loss_d": status.get('loss_d', 0.0),
            "eta_seconds": status.get('eta_seconds', 0),
            "message": status.get('message', ''),
            "scale_previews": status.get('scale_previews', [])
        }), 200
    
    except Exception as e:
        logger.error(f"Error in training_status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/generate-consingan', methods=['POST'])
def generate_consingan():
    """Generate synthetic images using trained ConSinGAN model."""
    try:
        import time
        start_time = time.time()
        
        data = request.json
        
        if not data or 'session_id' not in data:
            return jsonify({
                "success": False,
                "error": "Missing session_id"
            }), 400
        
        session_id = data['session_id']
        num_images = data.get('num_images', 5)
        
        if not isinstance(num_images, int) or num_images < 1 or num_images > 50:
            return jsonify({
                "success": False,
                "error": "num_images must be between 1 and 50"
            }), 400
        
        if not SessionManager.session_exists(session_id):
            return jsonify({
                "success": False,
                "error": f"Session {session_id} does not exist"
            }), 404
        
        try:
            device = DeviceManager.get_optimal_device()
            inference = ConSinGANInference(session_id, device)
            
            inference.load_model(session_id)
            
            generated_images = inference.generate_base64(num_images)
            
            generation_time = time.time() - start_time
            
            logger.info(f"Generated {num_images} images for session {session_id} in {generation_time:.2f}s")
            
            return jsonify({
                "success": True,
                "images": generated_images,
                "metadata": {
                    "model_version": "1.0",
                    "generation_time": generation_time,
                    "device": device
                }
            }), 200
        
        except FileNotFoundError as e:
            logger.error(f"Model not found for session {session_id}: {e}")
            return jsonify({
                "success": False,
                "error": "Model not trained yet. Please train the model first."
            }), 404
        
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    except Exception as e:
        logger.error(f"Error in generate_consingan: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --- Model Library Management (Updated) ---
        
        return jsonify({
            "success": True,
            "message": f"Model {model_id} deleted"
        }), 200
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/generate-from-model/<model_id>', methods=['POST'])
def generate_from_model(model_id):
    """Generate images using a trained model from the model library."""
    try:
        import time
        start_time = time.time()

        req_data   = request.json or {}
        num_images = req_data.get('num_images', 5)

        if not isinstance(num_images, int) or num_images < 1 or num_images > 50:
            return jsonify({
                "success": False,
                "error": "num_images must be between 1 and 50"
            }), 400

        model_data = model_manager.get_model(model_id)
        if not model_data:
            return jsonify({
                "success": False,
                "error": f"Model {model_id} not found"
            }), 404

        try:
            device = DeviceManager.get_optimal_device()

            # model_library klasöründe config.json değil metadata.json var.
            # config bilgisi metadata.json içindeki 'config' anahtarında.
            model_dir   = os.path.dirname(model_data['generator_path'])
            meta_path   = os.path.join(model_dir, "metadata.json")
            config_path = os.path.join(model_dir, "config.json")

            import json as _json
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = _json.load(f)
            elif os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = _json.load(f)
                config = meta.get('config', {})
            else:
                # Fallback: model_data içindeki config'i kullan
                config = model_data.get('config', {})

            num_scales  = config.get('num_scales', 1)
            scales      = [tuple(s) for s in config.get('scales', [])]
            noise_amps  = config.get('noise_amps', [1.0] * num_scales)

            # Tüm ölçeklerin G'lerini yükle
            generators = []
            for idx in range(num_scales):
                g_path = os.path.join(model_dir, f"generator_{idx}.pth")
                if not os.path.exists(g_path):
                    # Eski tek-G formatı: sadece generator.pth
                    g_path = model_data['generator_path']
                    g = ConSinGANGenerator(num_filters=32, num_blocks=5).to(device)
                    state = torch.load(g_path, map_location=device, weights_only=False)
                    g.load_state_dict(state)
                    g.eval()
                    generators = [g]
                    scales     = scales or [(256, 256)]
                    noise_amps = noise_amps[:1] or [1.0]
                    break
                g = ConSinGANGenerator(num_filters=32, num_blocks=5).to(device)
                state = torch.load(g_path, map_location=device, weights_only=False)
                g.load_state_dict(state)
                g.eval()
                generators.append(g)

            # z_opts varsa yükle
            z_opts = []
            z_path = os.path.join(model_dir, "z_opts.pth")
            if os.path.exists(z_path):
                z_list = torch.load(z_path, map_location=device, weights_only=False)
                z_opts = [z.to(device) for z in z_list]

            use_z_opts = len(z_opts) == len(generators)

            # Zincir üretim
            generated_images = []
            with torch.no_grad():
                for _ in range(num_images):
                    prev = None
                    for idx, g in enumerate(generators):
                        amp = noise_amps[idx] if idx < len(noise_amps) else 1.0

                        if scales and idx < len(scales):
                            h_i, w_i = scales[idx]
                        elif use_z_opts:
                            h_i, w_i = z_opts[idx].shape[-2:]
                        else:
                            h_i = w_i = 256

                        if use_z_opts:
                            perturb = 0.1 * amp * torch.randn(1, 3, h_i, w_i, device=device)
                            noise_i = z_opts[idx] + perturb
                        else:
                            noise_i = amp * torch.randn(1, 3, h_i, w_i, device=device)

                        prev = g(noise_i, prev)

                        if idx < len(generators) - 1:
                            if scales and (idx + 1) < len(scales):
                                next_h, next_w = scales[idx + 1]
                            else:
                                next_h = next_w = prev.shape[-1] * 2
                            prev = torch.nn.functional.interpolate(
                                prev, size=(next_h, next_w),
                                mode="bilinear", align_corners=False
                            )

                    arr = prev.squeeze(0).cpu().numpy()
                    arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    arr = np.transpose(arr, (1, 2, 0))

                    from PIL import Image as _PIL
                    from io import BytesIO as _BytesIO
                    buf = _BytesIO()
                    _PIL.fromarray(arr, mode='RGB').save(buf, format='JPEG', quality=95)
                    buf.seek(0)
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    generated_images.append({
                        'image': f"data:image/jpeg;base64,{b64}",
                        'label': 'Sentetik (AI)',
                        'ssim':  0.0
                    })

            generation_time = time.time() - start_time
            logger.info(
                f"Generated {num_images} images from model {model_id} "
                f"in {generation_time:.2f}s"
            )
            return jsonify({
                "success": True,
                "images":  generated_images,
                "metadata": {
                    "model_id":        model_id,
                    "model_name":      model_data['model_name'],
                    "generation_time": generation_time,
                    "device":          device
                }
            }), 200

        except Exception as e:
            logger.error(f"Error generating images: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error in generate_from_model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# --- Model Library Management ---

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all trained models in the library."""
    try:
        models = model_manager.list_models()
        
        # Format model data for frontend
        formatted_models = []
        for model in models:
            formatted_models.append({
                'model_id': model.get('model_id'),
                'model_name': model.get('model_name'),
                'created_date': model.get('created_date'),
                'description': model.get('description', ''),
                'has_reference_image': os.path.exists(model.get('reference_image_path', ''))
            })
        
        return jsonify({
            "success": True,
            "models": formatted_models,
            "count": len(formatted_models)
        }), 200
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get details of a specific model."""
    try:
        model = model_manager.get_model(model_id)
        
        if not model:
            return jsonify({
                "success": False,
                "error": f"Model not found: {model_id}"
            }), 404
        
        return jsonify({
            "success": True,
            "model": {
                'model_id': model.get('model_id'),
                'model_name': model.get('model_name'),
                'created_date': model.get('created_date'),
                'description': model.get('description', ''),
                'config': model.get('config', {}),
                'has_reference_image': os.path.exists(model.get('reference_image_path', ''))
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/models/<model_id>/rename', methods=['POST'])
def rename_model(model_id):
    """Rename a model."""
    try:
        data = request.json
        
        if not data or 'new_name' not in data:
            return jsonify({
                "success": False,
                "error": "Missing new_name parameter"
            }), 400
        
        new_name = data['new_name']
        new_model_id = model_manager.rename_model(model_id, new_name)
        
        if not new_model_id:
            return jsonify({
                "success": False,
                "error": f"Failed to rename model {model_id}"
            }), 500
        
        return jsonify({
            "success": True,
            "new_model_id": new_model_id,
            "message": f"Model renamed to {new_name}"
        }), 200
    
    except Exception as e:
        logger.error(f"Error renaming model {model_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model from the library."""
    try:
        success = model_manager.delete_model(model_id)
        
        if not success:
            return jsonify({
                "success": False,
                "error": f"Failed to delete model {model_id}"
            }), 500
        
        return jsonify({
            "success": True,
            "message": f"Model {model_id} deleted successfully"
        }), 200
    
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/models/<model_id>/reference-image', methods=['GET'])
def get_reference_image(model_id):
    """Get the reference image for a model."""
    try:
        model = model_manager.get_model(model_id)
        
        if not model:
            return jsonify({
                "success": False,
                "error": f"Model not found: {model_id}"
            }), 404
        
        ref_image_path = model.get('reference_image_path')
        
        if not os.path.exists(ref_image_path):
            return jsonify({
                "success": False,
                "error": "Reference image not found"
            }), 404
        
        return send_from_directory(os.path.dirname(ref_image_path), os.path.basename(ref_image_path))
    
    except Exception as e:
        logger.error(f"Error getting reference image for model {model_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --- Frontend Routes (MUST BE LAST) ---

@app.route('/')
def serve_index():
    """Serve the main index.html file"""
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory(BASE_DIR, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Flask Backend Başlatılıyor... | Port: {port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    finally:
        task_manager.shutdown()
