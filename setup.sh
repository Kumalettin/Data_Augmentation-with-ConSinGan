#!/bin/bash

# ConSinGAN Sanal Ortam Kurulum Scripti

echo "🚀 ConSinGAN Sanal Ortamı Kurulumu Başlıyor..."

# 1. Sanal ortam oluştur
echo "📦 Sanal ortam oluşturuluyor..."
python -m venv venv

# 2. Sanal ortamı aktif et
echo "✅ Sanal ortam aktif ediliyor..."
source venv/Scripts/activate

# 3. Pip'i güncelle
echo "🔄 Pip güncelleniyor..."
pip install --upgrade pip setuptools wheel

# 4. Bağımlılıkları yükle
echo "📥 Bağımlılıklar yükleniyor (bu biraz zaman alabilir)..."
pip install -r requirements.txt

# 5. Test bağımlılıkları
echo "🧪 Test bağımlılıkları yükleniyor..."
pip install pytest pytest-flask flask-cors

# 6. Kurulumu doğrula
echo "✔️ Kurulum doğrulanıyor..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} yüklendi'); print(f'GPU Desteği: {torch.cuda.is_available()}')"
python -c "import cv2, flask, numpy; print('✅ Tüm kütüphaneler başarıyla yüklendi!')"

echo ""
echo "🎉 Kurulum tamamlandı!"
echo ""
echo "📝 Sonraki adımlar:"
echo "1. Sanal ortamı aktif et: source venv/Scripts/activate"
echo "2. Backend'i çalıştır: python backend.py"
echo "3. Tarayıcıda aç: http://localhost:5000"
echo ""
