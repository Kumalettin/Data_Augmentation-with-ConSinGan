# 🏛️ Data Augmentation AI — ConSinGAN Tarihi Eser ve Anıt Veri Çoğaltma Sistemi

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-000000.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Data Augmentation AI**, kültürel mirasın korunması, tarihi eser/anıt sınıflandırması ve mimari görüntü analizi çalışmaları için **tek bir tarihi eser veya anıt fotoğrafından** yüksek kaliteli ve çeşitli sentetik veri kümeleri üretmek amacıyla geliştirilmiş uçtan uca bir web uygulamasıdır. 

Uygulama, **Geleneksel Veri Çoğaltma (Traditional Data Augmentation)** teknikleri ile **ConSinGAN (Concurrent Single-Image GAN)** tabanlı derin öğrenme mimarisini tek bir modern arayüzde birleştirir.

---

## ✨ Temel Özellikler

- 🖼️ **Çift Modlu Veri Çoğaltma:**
  - **Geleneksel Augmentation:** Döndürme, Yatay/Dikey Aynalama, Parlaklık, Kontrast, Gaussian Bulanıklık ve Gürültü ekleme.
  - **AI Destekli Augmentation (ConSinGAN):** Tek bir eğitim görselinden tarihi eserin/anıtın genel mimari formunu, sütun/kemer dizilimini ve taş yüzey dokusunu öğrenerek sıfırdan özgün sentetik görseller üretme.
- 🧠 **Orijinal Makaleye Uyumlu ConSinGAN Mimarisi:**
  - Dinamik çözünürlük piramidi (`25px` → orijinal boyut, `0.75` ölçek çarpanı).
  - **Residual Generator:** Her ölçekte yalnızca mimari detay ve doku farklarını (`prev_img + out`) öğrenen 5 katmanlı `ConvBlock` yapısı.
  - **PatchGAN Discriminator:** `3x3` konvolüsyonlarla (`stride=1`) her görüntü yamasını (patch) ayrı değerlendiren yerel doku denetçisi.
  - **Z_opt & Eşzamanlı Eğitim (Concurrent Training):** Sabit gürültü (`z_opt`) ile yeniden yapılandırma (L1) yolunu ve rastgele gürültü ile çekişmeli (Adversarial) yolu ayırarak kademeli öğrenme hızıyla (`lr * 0.1^k`) çok ölçekli eğitim.
- 📚 **Kalıcı Model Kütüphanesi (`model_library/`):**
  - Eğitilen tarihi eser/anıt modellerini isimlendirerek kaydetme, listeleme, yeniden adlandırma ve silme.
  - **Hazır Model ile Hızlı Üretim:** Kayıtlı modellerden yeniden eğitim beklemeksizin saniyeler içinde yeni sentetik görüntüler üretme.
- 📊 **Canlı Eğitim Takibi & Ara Katman Önizleme:**
  - Arka plan iş parçacıkları (`TrainingTaskManager`) sayesinde arayüzü dondurmayan asenkron eğitim.
  - Gerçek zamanlı ilerleme çubuğu (`%`), Generator/Discriminator kayıp (loss) değerleri, tahmini kalan süre (`ETA`) ve her ölçek tamamlandığında üretilen ara katman (`Scale 1, Scale 2...`) önizlemeleri.
- 🔬 **Doku Koruma ve Kalite Ölçümü (SSIM):**
  - Üretilen görüntülerin orijinal tarihi eserin taş işçiliğine ve mimari hatlarına yapısal benzerliğini **SSIM (Structural Similarity Index)**, Sobel kenar gradyanları ve yerel kontrast analizi ile değerlendirme.
- ⚡ **Otomatik GPU / CPU Yönetimi:**
  - NVIDIA CUDA destekli ekran kartlarını otomatik algılama, bellek taşması (OOM) durumunda önbellek temizleme ve CPU'ya güvenli geçiş.

---

## 🗂️ Proje Dosya Yapısı

```text
data_aug/
├── backend.py                  # Flask REST API sunucusu ve ana yönlendirici
├── models.py                   # ConSinGANGenerator, ConSinGANDiscriminator (PatchGAN) ve ConSinGANLoss
├── training.py                 # ConSinGANTrainer (çok ölçekli eğitim) ve TrainingTaskManager (arka plan iş havuzu)
├── inference.py                # ConSinGANInference (çok ölçekli zincirleme sentetik görüntü üretimi)
├── augmentation.py             # Format-bağımsız (OpenCV/PIL) geleneksel veri çoğaltma modülü
├── texture_preservation.py     # SSIM ve Sobel/kontrast tabanlı mimari ve yüzey dokusu koruma modülü
├── model_manager.py            # Kalıcı model kütüphanesi (model_library/) kayıt ve yönetim sınıfı
├── session_manager.py          # UUID tabanlı kullanıcı oturumu ve geçici dosya izolasyonu
├── device_manager.py           # GPU (CUDA) / CPU algılama ve bellek yönetimi
├── index.html                  # Modern, çift sekmeli web kullanıcı arayüzü
├── styles.css                  # Glassmorphism ve koyu tema (Dark Mode) tasarım stilleri
├── app.js                      # İstemci tarafı mantığı, Canvas işlemleri ve API iletişimi
├── requirements.txt            # Python kütüphane bağımlılıkları
├── setup.sh                    # Sanal ortam ve bağımlılık kurulum betiği
└── tests/                      # Pytest ve Hypothesis tabanlı özellik (Property-Based) testleri
    └── properties/
        ├── test_api_endpoints.py
        ├── test_augmentation.py
        ├── test_gpu_memory_management.py
        ├── test_inference.py
        ├── test_models.py
        ├── test_pytorch_loading.py
        ├── test_texture_preservation.py
        └── test_training.py
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Ön Gereksinimler
- **Python:** `3.9` veya üzeri
- **Donanım:** Eğitim hızı için NVIDIA CUDA destekli GPU önerilir (GPU yoksa sistem otomatik olarak CPU modunda çalışır).

### 2. Projeyi Klonlayın ve Sanal Ortam Oluşturun

```bash
git clone <repo-url>
cd data_aug

# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı aktifleştirme (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Sanal ortamı aktifleştirme (Linux / macOS / Git Bash)
source venv/Scripts/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **İpucu (GPU Desteği):** NVIDIA ekran kartınız varsa PyTorch'un CUDA sürümünü [pytorch.org](https://pytorch.org/get-started/locally/) adresinden sisteminize uygun olarak yükleyebilirsiniz.

### 4. Uygulamayı Başlatın

```bash
python backend.py
```

Sunucu başladığında tarayıcınızdan şu adrese gidin:
👉 **`http://localhost:5000`**

---

## 📖 Kullanım Kılavuzu

### 🔄 1. Geleneksel Augmentation Sekmesi
1. **Görsel Yükle:** Sürükle-bırak alanına bir tarihi eser veya anıt fotoğrafı (`PNG`, `JPG`, `JPEG`, maks. `10MB`) bırakın.
2. **Ayarları Seçin:** Üretilecek görsel sayısını (`1-50`) ve uygulanacak teknikleri (*Döndürme, Aynalama, Parlaklık, Kontrast, Bulanıklık, Gürültü*) işaretleyin.
3. **Oluştur ve İndir:** **"✨ Görselleri Oluştur"** butonuna tıklayın. Üretilen görselleri tek tek veya **"📥 Hepsini İndir"** butonuyla topluca indirin.

### 🤖 2. AI Destekli Augmentation (ConSinGAN) Sekmesi

#### A) Yeni Model Eğit Modu
1. **"🎓 Yeni Model Eğit"** modunu seçin ve eğitim yapılacak tarihi eser/anıt fotoğrafını yükleyin.
2. **Model Adı (Opsiyonel):** Modeli daha sonra tekrar kullanmak istiyorsanız bir isim girin (örn. `efes_antik_anıt_v1`).
3. **Hiperparametreler:**
   - `Iteration Sayısı`: `100` – `10000` (Hızlı test için `500`, dengeli sonuç için `2000` önerilir).
   - `Learning Rate`: `0.00001` – `0.001` (Varsayılan: `0.0002`).
4. **"🎓 Modeli Eğit"** butonuna tıklayın. İlerleme çubuğundan eğitim yüzdesini ve her ölçek sonunda oluşan ara katman görüntülerini takip edin.
5. Eğitim tamamlandığında **"🎨 Sentetik Görsel Üret"** butonuna tıklayarak yeni sentetik tarihi eser/anıt görselleri üretin.

#### B) Hazır Model Kullan Modu
1. **"⚡ Hazır Model Kullan"** moduna geçin.
2. Açılır listeden daha önce eğittiğiniz bir modeli seçin ve üretilecek görsel sayısını belirleyin.
3. **"⚡ Hızlı Üret"** butonuna tıklayarak saniyeler içinde yeni sentetik görseller elde edin.

---

## 🔌 REST API Uç Noktaları (Endpoints)

| Metot | Uç Nokta (Endpoint) | Açıklama |
| :--- | :--- | :--- |
| `POST` | `/api/augment` | Seçilen geleneksel tekniklerle görsel çoğaltma yapar |
| `POST` | `/api/session` | Yeni bir kullanıcı oturumu (`UUID`) oluşturur |
| `POST` | `/api/upload` | Oturuma ait orijinal eğitim görselini yükler |
| `POST` | `/api/train-consingan` | Arka planda asenkron ConSinGAN eğitimini başlatır |
| `GET` | `/api/training-status/<task_id>` | Eğitim ilerlemesini, kayıp değerlerini ve ölçek önizlemelerini döner |
| `POST` | `/api/generate-consingan` | Aktif oturumda eğitilmiş modelden sentetik görseller üretir |
| `GET` | `/api/models` | Kütüphanedeki (`model_library/`) tüm kayıtlı modelleri listeler |
| `GET` | `/api/models/<model_id>` | Belirli bir modelin meta verilerini ve ayarlarını döner |
| `POST` | `/api/models/<model_id>/rename` | Kayıtlı bir modeli yeniden adlandırır |
| `DELETE` | `/api/models/<model_id>` | Kayıtlı bir modeli kütüphaneden siler |
| `POST` | `/api/generate-from-model/<model_id>` | Kütüphanedeki hazır bir modelden doğrudan sentetik görsel üretir |

---

## 🧪 Testlerin Çalıştırılması

Proje, `pytest` ve `hypothesis` kullanılarak yazılmış kapsamlı **Özellik Tabanlı Test (Property-Based Testing)** paketine sahiptir:

```bash
pytest tests/ -v
```

Discriminator mimarisini hızlıca doğrulamak için:

```bash
python test_discriminator_fix.py
```

---

## 📄 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Lisans tezi çalışması kapsamında geliştirilmiştir.
