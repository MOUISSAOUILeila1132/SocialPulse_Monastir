document.addEventListener('DOMContentLoaded', function() {
    // Sections de la page
    const homeSection = document.getElementById('home-section');
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');

    // Éléments interactifs
    const startAnalysisBtn = document.getElementById('start-analysis-btn');
    const uploadForm = document.getElementById('video-upload-form');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video_file');
    const loadingMessage = document.getElementById('loading-message');
    const statusText = document.getElementById('status-text');
    const videoPreview = document.getElementById('video-preview');

    // Affiche la section d'accueil par défaut
    homeSection.classList.add('active');

    // Gère la navigation de l'accueil à l'upload
    if (startAnalysisBtn) {
        startAnalysisBtn.addEventListener('click', () => {
            homeSection.classList.remove('active');
            uploadSection.classList.add('active');
        });
    }

    // Gère la sélection de fichier
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });
    }

    function handleFile(file) {
        // Affiche le lecteur vidéo et y charge la vidéo
        const videoURL = URL.createObjectURL(file);
        videoPreview.src = videoURL;
        videoPreview.style.display = 'block';
        videoPreview.play();

        // Met à jour l'interface
        uploadForm.style.display = 'none';
        loadingMessage.style.display = 'block';
        statusText.textContent = 'Téléversement de la vidéo...';

        // Envoie le fichier au serveur en arrière-plan
        const formData = new FormData();
        formData.append('video_file', file);

        fetch('/start-analysis', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Erreur serveur') });
            }
            return response.json();
        })
        .then(data => {
            if (data.task_id) {
                statusText.textContent = 'Analyse en cours...';
                pollStatus(data.task_id);
            } else {
                 throw new Error(data.error || 'ID de tâche non reçu.');
            }
        })
        .catch(error => {
            statusText.textContent = `Erreur : ${error.message}`;
        });
    }

    function pollStatus(taskId) {
        const interval = setInterval(() => {
            fetch(`/status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.state === 'SUCCESS') {
                        clearInterval(interval);
                        displayResults(data.result);
                    } else if (data.state === 'FAILURE') {
                        clearInterval(interval);
                        statusText.textContent = `L'analyse a échoué. Veuillez réessayer.`;
                        console.error("Erreur de la tâche Celery:", data.status);
                    } else {
                        statusText.textContent = `Analyse en cours... (${data.status})`;
                    }
                })
                .catch(err => {
                    clearInterval(interval);
                    statusText.textContent = 'Erreur de connexion. Impossible de vérifier le statut.';
                });
        }, 3000); // Vérifie toutes les 3 secondes
    }

    function displayResults(results) {
        uploadSection.classList.remove('active');
        resultsSection.classList.add('active');

        // Récupération sécurisée du résumé (si disponible dans results.video_summary)
        const summaryText = results.video_summary || "Aucun résumé généré pour cette vidéo.";

        const resultsHTML = `
            <div class="results-dashboard">
                <div class="dashboard-header">
                    <div>
                        <i class="fas fa-chart-bar"></i>
                        <h2>AI Analysis Complete</h2>
                        <p>Tableau de bord complet de l'intelligence vidéo</p>
                    </div>
                    <div class="header-right">
                        <div class="accuracy-badge">${results.topic_confidence || 'N/A'} Accuracy</div>
                        <a href="/" class="analyze-new-btn"><i class="fas fa-upload"></i> Analyze New Video</a>
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-icon"><i class="far fa-clock"></i></div>
                        <div class="metric-info"><span>Duration</span><strong>${results.duration_seconds}s</strong></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon"><i class="fas fa-desktop"></i></div>
                        <div class="metric-info"><span>Resolution</span><strong>${results.resolution || 'N/A'}</strong></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon"><i class="fas fa-film"></i></div>
                        <div class="metric-info"><span>FPS</span><strong>${results.fps || 'N/A'}</strong></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon"><i class="far fa-user"></i></div>
                        <div class="metric-info"><span>Faces Detected</span><strong>${results.face_detection}</strong></div>
                    </div>
                </div>

                <!-- SECTION AJOUTÉE : RÉSUMÉ VIDÉO -->
                <div class="summary-card">
                    <h3><i class="fas fa-magic"></i> AI Video Summary</h3>
                    <div class="card-body">
                        <p>${summaryText}</p>
                    </div>
                </div>

                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h3><i class="fas fa-microphone-alt"></i> Speech Analysis</h3>
                        <div class="speech-status">
                            <span>Speech Detected</span>
                            ${results.has_speech 
                                ? `<span class="status-yes"><i class="fas fa-check-circle"></i> Yes</span>` 
                                : `<span class="status-no"><i class="fas fa-times-circle"></i> No</span>`}
                        </div>
                        <div class="keywords">
                            <span>Detected Topic</span>
                            <div class="tags"><span class="tag">${results.topic}</span></div>
                        </div>
                    </div>

                    <div class="analysis-card">
                        <h3><i class="far fa-smile"></i> Sentiment Analysis</h3>
                        <div class="sentiment-item">
                            <span>${results.emotion}</span>
                            <div class="progress-bar"><div class="progress" style="width: ${results.emotion_confidence || '0%'};"></div></div>
                        </div>
                    </div>
                </div>
                
                <div class="transcription-card">
                     <h3><i class="far fa-file-alt"></i> AI Transcription</h3>
                     <div class="transcription-actions">
                         <button class="action-btn"><i class="far fa-copy"></i> Copy Text</button>
                         <button class="action-btn export-btn"><i class="fas fa-download"></i> Export Report</button>
                     </div>
                     <div class="transcription-content">
                        <p>${results.transcribed_text}</p>
                     </div>
                     <div class="transcription-stats">
                        <div class="stat-item"><strong>-</strong><span>Words</span></div>
                        <div class="stat-item"><strong>-</strong><span>Characters</span></div>
                        <div class="stat-item"><strong>-</strong><span>Accuracy</span></div>
                     </div>
                </div>
            </div>
        `;
        resultsSection.innerHTML = resultsHTML;
    }
});