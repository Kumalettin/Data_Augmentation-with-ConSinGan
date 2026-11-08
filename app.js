document.addEventListener('DOMContentLoaded', () => {
    // --- Tab Management ---
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            button.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
            localStorage.setItem('activeTab', tabName);
        });
    });
    
    const savedTab = localStorage.getItem('activeTab') || 'traditional';
    const savedTabButton = document.querySelector(`[data-tab="${savedTab}"]`);
    if (savedTabButton) savedTabButton.click();

    // --- DOM Elements (Traditional Tab) ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const originalImage = document.getElementById('original-image');
    const removeImageBtn = document.getElementById('remove-image');
    const settingsSection = document.getElementById('settings-section');
    const generateBtn = document.getElementById('generate-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsActions = document.getElementById('results-actions');
    const resultsCount = document.getElementById('results-count');
    const resultsGrid = document.getElementById('results-grid');
    const downloadAllBtn = document.getElementById('download-all-btn');

    // --- DOM Elements (AI Tab) ---
    const dropZoneAI = document.getElementById('drop-zone-ai');
    const fileInputAI = document.getElementById('file-input-ai');
    const previewContainerAI = document.getElementById('preview-container-ai');
    const originalImageAI = document.getElementById('original-image-ai');
    const removeImageBtnAI = document.getElementById('remove-image-ai');
    const settingsSectionAI = document.getElementById('settings-section-ai');
    const trainButton = document.getElementById('train-button');
    const iterationsInput = document.getElementById('iterations');
    const learningRateInput = document.getElementById('learning-rate');
    const trainingProgressContainer = document.getElementById('training-progress-container');
    const trainingProgressBar = document.getElementById('training-progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressETA = document.getElementById('progress-eta');
    const trainingStatusText = document.getElementById('training-status-text');
    const trainingError = document.getElementById('training-error');
    const resultsGridAI = document.getElementById('results-grid-ai');
    const resultsActionsAI = document.getElementById('results-actions-ai');
    const resultsCountAI = document.getElementById('results-count-ai');
    const downloadAllBtnAI = document.getElementById('download-all-btn-ai');

    const MAX_FILE_SIZE = 10 * 1024 * 1024;
    const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];

    // State
    let currentImageDataUrl = null;
    let generatedImagesData = [];
    let imgElement = new Image();
    let currentImageDataUrlAI = null;
    let currentSessionId = null;
    let currentTaskId = null;
    let trainingInProgress = false;
    let generatedImagesDataAI = [];

    // --- Traditional Tab: Drag & Drop ---
    dropZone.addEventListener('click', () => { if (!currentImageDataUrl) fileInput.click(); });

    ['dragover', 'dragenter'].forEach(e => dropZone.addEventListener(e, ev => {
        ev.preventDefault(); ev.stopPropagation();
        if (!currentImageDataUrl) dropZone.classList.add('dragover');
    }));
    ['dragleave', 'dragend', 'drop'].forEach(e => dropZone.addEventListener(e, ev => {
        ev.preventDefault(); ev.stopPropagation();
        dropZone.classList.remove('dragover');
    }));
    dropZone.addEventListener('drop', e => { if (!currentImageDataUrl && e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
    fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });
    removeImageBtn.addEventListener('click', e => { e.stopPropagation(); resetUploader(); });

    // --- AI Tab: Drag & Drop ---
    dropZoneAI.addEventListener('click', () => { if (!currentImageDataUrlAI) fileInputAI.click(); });

    ['dragover', 'dragenter'].forEach(e => dropZoneAI.addEventListener(e, ev => {
        ev.preventDefault(); ev.stopPropagation();
        if (!currentImageDataUrlAI) dropZoneAI.classList.add('dragover');
    }));
    ['dragleave', 'dragend', 'drop'].forEach(e => dropZoneAI.addEventListener(e, ev => {
        ev.preventDefault(); ev.stopPropagation();
        dropZoneAI.classList.remove('dragover');
    }));
    dropZoneAI.addEventListener('drop', e => { if (!currentImageDataUrlAI && e.dataTransfer.files.length) handleFileAI(e.dataTransfer.files[0]); });
    fileInputAI.addEventListener('change', e => { if (e.target.files.length) handleFileAI(e.target.files[0]); });
    removeImageBtnAI.addEventListener('click', e => { e.stopPropagation(); resetUploaderAI(); });

    // --- Hyperparameter Validation ---
    function validateHyperparameters() {
        let isValid = true;
        const iterations = parseInt(iterationsInput.value);
        const iterationsError = document.getElementById('iterations-error');
        if (isNaN(iterations) || iterations < 100 || iterations > 10000) {
            iterationsError.textContent = 'Iterations must be between 100 and 10000';
            iterationsError.style.display = 'block';
            isValid = false;
        } else {
            iterationsError.style.display = 'none';
        }
        const learningRate = parseFloat(learningRateInput.value);
        const learningRateError = document.getElementById('learning-rate-error');
        if (isNaN(learningRate) || learningRate < 0.00001 || learningRate > 0.001) {
            learningRateError.textContent = 'Learning rate must be between 0.00001 and 0.001';
            learningRateError.style.display = 'block';
            isValid = false;
        } else {
            learningRateError.style.display = 'none';
        }
        return isValid;
    }

    iterationsInput.addEventListener('change', validateHyperparameters);
    learningRateInput.addEventListener('change', validateHyperparameters);

    // --- AI Tab: Training ---
    trainButton.addEventListener('click', async () => {
        if (!currentImageDataUrlAI) { alert('Lütfen önce bir görsel yükleyin'); return; }
        if (!validateHyperparameters()) return;

        trainButton.disabled = true;
        trainingError.style.display = 'none';
        trainingProgressContainer.style.display = 'block';
        trainingInProgress = true;

        try {
            const sessionResponse = await fetch('/api/session', { method: 'POST' });
            const sessionData = await sessionResponse.json();
            currentSessionId = sessionData.session_id;

            const base64Image = currentImageDataUrlAI.split(',')[1];
            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, image: base64Image })
            });
            if (!uploadResponse.ok) throw new Error('Image upload failed');

            const modelNameInput = document.getElementById('model-name');
            const modelName = modelNameInput ? modelNameInput.value.trim() || null : null;

            const trainResponse = await fetch('/api/train-consingan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    iterations: parseInt(iterationsInput.value),
                    learning_rate: parseFloat(learningRateInput.value),
                    batch_size: 1,
                    num_scales: 6,
                    model_name: modelName
                })
            });

            const trainData = await trainResponse.json();
            if (!trainData.success) throw new Error(trainData.error || 'Training failed');

            currentTaskId = trainData.task_id;
            await pollTrainingStatus();

        } catch (error) {
            console.error('Training error:', error);
            trainingError.textContent = `Error: ${error.message}`;
            trainingError.style.display = 'block';
        } finally {
            trainButton.disabled = false;
            trainingInProgress = false;
        }
    });

    // --- Poll Training Status ---
    async function pollTrainingStatus() {
        const maxAttempts = 1000;
        let attempts = 0;
        const intermediateGallery = document.getElementById('intermediate-gallery');

        while (attempts < maxAttempts && trainingInProgress) {
            try {
                const response = await fetch(`/api/training-status/${currentTaskId}`);
                const data = await response.json();

                trainingProgressBar.value = data.progress;
                progressPercentage.textContent = `${data.progress}%`;

                if (data.eta_seconds) {
                    const minutes = Math.floor(data.eta_seconds / 60);
                    const seconds = data.eta_seconds % 60;
                    progressETA.textContent = `ETA: ${minutes}m ${seconds}s`;
                }

                trainingStatusText.textContent = data.message || 'Training in progress...';

                // Ara katman görsellerini güncelle
                if (data.scale_previews && data.scale_previews.length > 0) {
                    data.scale_previews.forEach(preview => {
                        const existingEl = intermediateGallery.querySelector(`[data-scale="${preview.scale_num}"]`);
                        if (existingEl) {
                            existingEl.querySelector('img').src = preview.image;
                        } else {
                            const item = document.createElement('div');
                            item.dataset.scale = preview.scale_num;
                            item.style.cssText = 'text-align:center; font-size:0.75rem;';
                            item.innerHTML = `<img src="${preview.image}" alt="Scale ${preview.scale_num}" style="width:80px;height:80px;object-fit:cover;border-radius:6px;border:2px solid #4f46e5;"><br>Scale ${preview.scale_num}`;
                            intermediateGallery.appendChild(item);
                        }
                    });
                }

                if (data.status === 'completed') {
                    trainingProgressContainer.style.display = 'none';
                    
                    // Model kütüphanesini yenile
                    loadModelLibrary();
                    
                    // Üretim butonu ekle
                    const existingBtn = document.getElementById('generate-btn-ai');
                    if (!existingBtn) {
                        const generateBtnAI = document.createElement('button');
                        generateBtnAI.id = 'generate-btn-ai';
                        generateBtnAI.className = 'btn primary-btn';
                        generateBtnAI.style.marginTop = '1rem';
                        generateBtnAI.textContent = '🎨 Sentetik Görsel Üret';
                        generateBtnAI.addEventListener('click', generateSyntheticImages);
                        settingsSectionAI.appendChild(generateBtnAI);
                    }

                    alert('✅ Eğitim tamamlandı! Model kütüphaneye kaydedildi.');
                    break;
                } else if (data.status === 'failed') {
                    throw new Error(data.message || 'Training failed');
                }

                await new Promise(resolve => setTimeout(resolve, 2000));
                attempts++;

            } catch (error) {
                console.error('Status poll error:', error);
                trainingError.textContent = `Error: ${error.message}`;
                trainingError.style.display = 'block';
                break;
            }
        }
    }

    // --- Generate Synthetic Images (from current session) ---
    async function generateSyntheticImages() {
        try {
            const response = await fetch('/api/generate-consingan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, num_images: 5 })
            });

            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Generation failed');

            generatedImagesDataAI = [];
            resultsGridAI.innerHTML = '';

            data.images.forEach((img, index) => {
                const fileName = `synthetic_${Date.now()}_${index + 1}.jpg`;
                appendResultToGalleryAI(img.image, fileName, 'Sentetik (AI)', img.ssim);
                generatedImagesDataAI.push({ dataUrl: img.image, fileName });
            });

            resultsCountAI.textContent = `${generatedImagesDataAI.length} Sonuç`;
            resultsActionsAI.style.display = 'flex';

        } catch (error) {
            console.error('Generation error:', error);
            alert(`Error: ${error.message}`);
        }
    }

    // --- Model Library ---
    async function loadModelLibrary() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            const modelLibraryList = document.getElementById('model-library-list');
            if (!modelLibraryList) return;

            if (!data.success || data.models.length === 0) {
                modelLibraryList.innerHTML = '<div class="empty-state"><p>Henüz kaydedilmiş model yok</p></div>';
                return;
            }

            modelLibraryList.innerHTML = '';
            data.models.forEach(model => {
                const card = document.createElement('div');
                card.className = 'model-card';
                const createdDate = new Date(model.created_date).toLocaleString('tr-TR');
                card.innerHTML = `
                    <div class="model-header">
                        <h4>${model.model_name}</h4>
                        <span class="model-date">${createdDate}</span>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-small" onclick="useModel('${model.model_id}')">🎨 Kullan</button>
                        <button class="btn btn-small" onclick="deleteModel('${model.model_id}')">🗑️ Sil</button>
                    </div>
                `;
                modelLibraryList.appendChild(card);
            });
        } catch (error) {
            console.error('Error loading model library:', error);
        }
    }

    window.useModel = async function(modelId) {
        try {
            const modelResponse = await fetch(`/api/models/${modelId}`);
            const modelData = await modelResponse.json();
            if (!modelData.success) { alert('Model yüklenemedi'); return; }

            const generateResponse = await fetch(`/api/generate-from-model/${modelId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_images: 5 })
            });

            const generateData = await generateResponse.json();
            if (!generateData.success) { alert('Görsel üretimi başarısız: ' + generateData.error); return; }

            generatedImagesDataAI = [];
            resultsGridAI.innerHTML = '';

            generateData.images.forEach((img, index) => {
                const fileName = `generated_${modelId}_${Date.now()}_${index + 1}.jpg`;
                appendResultToGalleryAI(img.image, fileName, `${modelData.model.model_name} (Kaydedilmiş)`, img.ssim);
                generatedImagesDataAI.push({ dataUrl: img.image, fileName });
            });

            resultsCountAI.textContent = `${generatedImagesDataAI.length} Sonuç`;
            resultsActionsAI.style.display = 'flex';

        } catch (error) {
            console.error('Error using model:', error);
            alert('Hata: ' + error.message);
        }
    };

    window.deleteModel = async function(modelId) {
        if (!confirm('Bu modeli silmek istediğinizden emin misiniz?')) return;
        try {
            const response = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
            const data = await response.json();
            if (!data.success) { alert('Model silinemedi: ' + data.error); return; }
            loadModelLibrary();
        } catch (error) {
            console.error('Error deleting model:', error);
            alert('Hata: ' + error.message);
        }
    };

    // --- AI Tab: File Handling ---
    function handleFileAI(file) {
        if (!ALLOWED_TYPES.includes(file.type)) { alert('❌ Sadece PNG, JPG veya JPEG yükleyebilirsiniz.'); return; }
        if (file.size > MAX_FILE_SIZE) { alert('❌ Dosya boyutu 10MB\'dan büyük olamaz.'); return; }

        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageDataUrlAI = e.target.result;
            originalImageAI.src = currentImageDataUrlAI;
            previewContainerAI.style.display = 'flex';
            settingsSectionAI.style.opacity = '1';
            settingsSectionAI.style.pointerEvents = 'all';
            trainButton.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUploaderAI() {
        currentImageDataUrlAI = null;
        fileInputAI.value = '';
        previewContainerAI.style.display = 'none';
        originalImageAI.src = '';
        settingsSectionAI.style.opacity = '0.5';
        settingsSectionAI.style.pointerEvents = 'none';
        trainButton.disabled = true;
        clearResultsAI();
    }

    function clearResultsAI() {
        generatedImagesDataAI = [];
        resultsGridAI.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🤖</div>
                <p>Henüz görsel oluşturulmadı. Başlamak için bir görsel yükleyip modeli eğitin.</p>
            </div>`;
        resultsActionsAI.style.display = 'none';
    }

    function appendResultToGalleryAI(dataUrl, fileName, label, ssim) {
        const card = document.createElement('div');
        card.className = 'result-card fade-in';
        const ssimText = ssim ? `SSIM: ${ssim.toFixed(3)}` : '';
        card.innerHTML = `
            <div class="result-image-container">
                <img src="${dataUrl}" alt="Sentetik Görsel">
                <div class="result-overlay">
                    <button class="btn btn-small" onclick="downloadImage('${dataUrl}', '${fileName}')">⬇️ İndir</button>
                </div>
            </div>
            <div class="result-info">
                <div class="tech-tags">
                    <span class="tech-tag">${label}</span>
                    ${ssimText ? `<span class="tech-tag">${ssimText}</span>` : ''}
                </div>
            </div>`;
        resultsGridAI.appendChild(card);
    }

    downloadAllBtnAI.addEventListener('click', async () => {
        if (generatedImagesDataAI.length === 0) return;
        downloadAllBtnAI.textContent = '⏳ İndiriliyor...';
        const delay = ms => new Promise(res => setTimeout(res, ms));
        for (let i = 0; i < generatedImagesDataAI.length; i++) {
            window.downloadImage(generatedImagesDataAI[i].dataUrl, generatedImagesDataAI[i].fileName);
            if (i !== generatedImagesDataAI.length - 1) await delay(300);
        }
        downloadAllBtnAI.innerHTML = '📥 Hepsini İndir';
    });

    // --- Traditional Tab: File Handling ---
    function handleFile(file) {
        if (!ALLOWED_TYPES.includes(file.type)) { alert('❌ Sadece PNG, JPG veya JPEG yükleyebilirsiniz.'); return; }
        if (file.size > MAX_FILE_SIZE) { alert('❌ Dosya boyutu 10MB\'dan büyük olamaz.'); return; }

        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageDataUrl = e.target.result;
            originalImage.src = currentImageDataUrl;
            previewContainer.style.display = 'flex';
            settingsSection.style.opacity = '1';
            settingsSection.style.pointerEvents = 'all';
            generateBtn.disabled = false;
            imgElement.src = currentImageDataUrl;
        };
        reader.readAsDataURL(file);
    }

    function resetUploader() {
        currentImageDataUrl = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        originalImage.src = '';
        settingsSection.style.opacity = '0.5';
        settingsSection.style.pointerEvents = 'none';
        generateBtn.disabled = true;
        clearResults();
    }

    function clearResults() {
        generatedImagesData = [];
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🖼️</div>
                <p>Henüz görsel oluşturulmadı. Başlamak için bir görsel yükleyip ayarları yapılandırın.</p>
            </div>`;
        resultsActions.style.display = 'none';
    }

    // --- Traditional Augmentation ---
    const rand = (min, max) => Math.random() * (max - min) + min;

    generateBtn.addEventListener('click', async () => {
        const numImages = parseInt(document.getElementById('num-images').value);
        if (isNaN(numImages) || numImages < 1 || numImages > 50) {
            alert("Lütfen 1 ile 50 arasında geçerli bir sayı giriniz.");
            return;
        }

        const selectedTechniques = Array.from(document.querySelectorAll('input[name="technique"]:checked')).map(cb => cb.value);
        if (selectedTechniques.length === 0) { alert("Lütfen en az 1 teknik seçin!"); return; }

        if (numImages > 20) {
            if (!confirm("20'den fazla görsel üretiyorsunuz. Devam etmek istiyor musunuz?")) return;
        }

        generateBtn.disabled = true;
        generateBtn.style.display = 'none';
        loadingIndicator.classList.remove('hidden');
        resultsGrid.innerHTML = '';
        resultsActions.style.display = 'none';
        generatedImagesData = [];

        try {
            for (let i = 0; i < numImages; i++) {
                await new Promise(resolve => setTimeout(resolve, 50));

                const appliedTechniquesNames = [];
                let localTechniques = selectedTechniques.filter(() => Math.random() > 0.3);
                if (localTechniques.length === 0) localTechniques = [selectedTechniques[Math.floor(Math.random() * selectedTechniques.length)]];

                const resultCanvas = document.createElement('canvas');
                const ctx = resultCanvas.getContext('2d');
                resultCanvas.width = imgElement.width;
                resultCanvas.height = imgElement.height;

                let filterString = '';
                let p_scaleX = 1, p_scaleY = 1, p_angle = 0;

                localTechniques.forEach(tech => {
                    switch (tech) {
                        case 'rotate':
                            const degrees = rand(-45, 45);
                            p_angle = degrees * Math.PI / 180;
                            appliedTechniquesNames.push(`🔄 Döndürme (${degrees.toFixed(1)}°)`);
                            break;
                        case 'flip':
                            if (Math.random() > 0.5) { p_scaleX = -1; appliedTechniquesNames.push('🪞 Yatay Aynalama'); }
                            else { p_scaleY = -1; appliedTechniquesNames.push('🪞 Dikey Aynalama'); }
                            break;
                        case 'brightness':
                            const bFactor = rand(0.5, 1.5).toFixed(2);
                            filterString += ` brightness(${bFactor})`;
                            appliedTechniquesNames.push(`☀️ Parlaklık (${bFactor}x)`);
                            break;
                        case 'contrast':
                            const cFactor = rand(0.5, 1.5).toFixed(2);
                            filterString += ` contrast(${cFactor})`;
                            appliedTechniquesNames.push(`🎨 Kontrast (${cFactor}x)`);
                            break;
                        case 'blur':
                            const bPx = rand(1, 5).toFixed(1);
                            filterString += ` blur(${bPx}px)`;
                            appliedTechniquesNames.push(`🌫️ Bulanıklık (${bPx}px)`);
                            break;
                    }
                });

                ctx.translate(resultCanvas.width / 2, resultCanvas.height / 2);
                ctx.rotate(p_angle);
                ctx.scale(p_scaleX, p_scaleY);
                if (filterString.trim()) ctx.filter = filterString.trim();
                ctx.drawImage(imgElement, -resultCanvas.width / 2, -resultCanvas.height / 2, resultCanvas.width, resultCanvas.height);
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.filter = 'none';

                if (localTechniques.includes('noise')) {
                    const noiseRange = Math.floor(rand(10, 50));
                    const imgData = ctx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
                    const d = imgData.data;
                    for (let j = 0; j < d.length; j += 4) {
                        d[j]     = Math.min(255, Math.max(0, d[j]     + Math.floor(rand(-noiseRange, noiseRange))));
                        d[j + 1] = Math.min(255, Math.max(0, d[j + 1] + Math.floor(rand(-noiseRange, noiseRange))));
                        d[j + 2] = Math.min(255, Math.max(0, d[j + 2] + Math.floor(rand(-noiseRange, noiseRange))));
                    }
                    ctx.putImageData(imgData, 0, 0);
                    appliedTechniquesNames.push('📡 Gürültü');
                }

                const outDataUrl = resultCanvas.toDataURL('image/jpeg', 0.9);
                const outFileName = `augmented_${Date.now()}_${i + 1}.jpg`;
                appendResultToGallery(outDataUrl, outFileName, appliedTechniquesNames);
                generatedImagesData.push({ dataUrl: outDataUrl, fileName: outFileName });
            }

            resultsCount.textContent = `${generatedImagesData.length} Sonuç`;
            resultsActions.style.display = 'flex';

        } catch (error) {
            console.error("İşlem Hatası:", error);
            alert("❌ Bir hata oluştu. Lütfen tekrar deneyin.");
        } finally {
            loadingIndicator.classList.add('hidden');
            generateBtn.style.display = 'flex';
            generateBtn.disabled = false;
        }
    });

    function appendResultToGallery(dataUrl, fileName, techniquesArr) {
        const card = document.createElement('div');
        card.className = 'result-card fade-in';
        const tagsHtml = techniquesArr.map(t => `<span class="tech-tag">${t}</span>`).join('');
        card.innerHTML = `
            <div class="result-image-container">
                <img src="${dataUrl}" alt="Sonuç">
                <div class="result-overlay">
                    <button class="btn btn-small" onclick="downloadImage('${dataUrl}', '${fileName}')">⬇️ İndir</button>
                </div>
            </div>
            <div class="result-info">
                <div class="tech-tags">${tagsHtml}</div>
            </div>`;
        resultsGrid.appendChild(card);
    }

    window.downloadImage = function(dataUrl, fileName) {
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    downloadAllBtn.addEventListener('click', async () => {
        if (generatedImagesData.length === 0) return;
        downloadAllBtn.textContent = '⏳ İndiriliyor...';
        const delay = ms => new Promise(res => setTimeout(res, ms));
        for (let i = 0; i < generatedImagesData.length; i++) {
            window.downloadImage(generatedImagesData[i].dataUrl, generatedImagesData[i].fileName);
            if (i !== generatedImagesData.length - 1) await delay(300);
        }
        downloadAllBtn.innerHTML = '📥 Hepsini İndir';
    });

    // --- Init ---
    loadModelLibrary();

    const aiTabButton = document.querySelector('[data-tab="ai"]');
    if (aiTabButton) aiTabButton.addEventListener('click', loadModelLibrary);

    // --- AI Mode Toggle ---
    const modeTrainBtn = document.getElementById('mode-train-btn');
    const modeUseBtn = document.getElementById('mode-use-btn');
    const panelTrain = document.getElementById('panel-train');
    const panelUse = document.getElementById('panel-use');

    function switchMode(mode) {
        if (mode === 'train') {
            modeTrainBtn.classList.add('active');
            modeUseBtn.classList.remove('active');
            panelTrain.style.display = 'block';
            panelUse.style.display = 'none';
        } else {
            modeUseBtn.classList.add('active');
            modeTrainBtn.classList.remove('active');
            panelTrain.style.display = 'none';
            panelUse.style.display = 'block';
            loadModelDropdown();
        }
    }

    modeTrainBtn.addEventListener('click', () => switchMode('train'));
    modeUseBtn.addEventListener('click', () => switchMode('use'));

    // --- Model Dropdown ---
    async function loadModelDropdown() {
        const select = document.getElementById('model-select');
        const quickBtn = document.getElementById('quick-generate-btn');
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            select.innerHTML = '';

            if (!data.success || data.models.length === 0) {
                select.innerHTML = '<option value="">-- Kayıtlı model yok --</option>';
                quickBtn.disabled = true;
                return;
            }

            select.innerHTML = '<option value="">-- Model seçin --</option>';
            data.models.forEach(model => {
                const date = new Date(model.created_date).toLocaleDateString('tr-TR');
                const opt = document.createElement('option');
                opt.value = model.model_id;
                opt.textContent = `${model.model_name}  (${date})`;
                select.appendChild(opt);
            });

            quickBtn.disabled = true;
            select.addEventListener('change', () => {
                quickBtn.disabled = !select.value;
            });

        } catch (err) {
            select.innerHTML = '<option value="">-- Yüklenemedi --</option>';
            quickBtn.disabled = true;
        }
    }

    document.getElementById('refresh-models-btn').addEventListener('click', loadModelDropdown);

    // --- Quick Generate ---
    document.getElementById('quick-generate-btn').addEventListener('click', async () => {
        const modelId = document.getElementById('model-select').value;
        if (!modelId) return;

        const numImages = parseInt(document.getElementById('num-images-ai').value) || 5;
        const quickBtn = document.getElementById('quick-generate-btn');
        const loading = document.getElementById('quick-generate-loading');
        const errorEl = document.getElementById('quick-generate-error');

        quickBtn.disabled = true;
        loading.classList.remove('hidden');
        errorEl.style.display = 'none';
        resultsGridAI.innerHTML = '';

        try {
            const response = await fetch(`/api/generate-from-model/${modelId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_images: numImages })
            });

            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Üretim başarısız');

            generatedImagesDataAI = [];

            data.images.forEach((img, index) => {
                const fileName = `quick_${modelId}_${Date.now()}_${index + 1}.jpg`;
                appendResultToGalleryAI(img.image, fileName, 'Sentetik (Hazır Model)', img.ssim);
                generatedImagesDataAI.push({ dataUrl: img.image, fileName });
            });

            resultsCountAI.textContent = `${generatedImagesDataAI.length} Sonuç`;
            resultsActionsAI.style.display = 'flex';

        } catch (err) {
            errorEl.textContent = 'Hata: ' + err.message;
            errorEl.style.display = 'block';
        } finally {
            quickBtn.disabled = false;
            loading.classList.add('hidden');
        }
    });
});
