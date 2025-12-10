# tasks.py
from celery import Celery
from ultra_simple_video_analyzer_local import UltraSimpleVideoAnalyzer
import os

# Configuration de Celery (Redis)
celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configuration optionnelle pour la stabilité sur Windows
celery.conf.update(
    CELERY_BROKER_TRANSPORT_OPTIONS={'visibility_timeout': 3600},
    CELERY_RESULT_EXTENDED=True
)

# Initialiser l'analyseur
analyzer = UltraSimpleVideoAnalyzer()

@celery.task(bind=True)
def analyze_video_task(self, video_path):
    try:
        # 1. Lancer l'analyse (résultat complexe imbriqué)
        raw_results = analyzer.analyze_video_ultra_simple(video_path)

        # Vérification d'erreur immédiate
        if "error" in raw_results:
            return {"error": raw_results["error"]}

        # 2. Extraire les données
        info = raw_results.get('video_info', {})
        analysis = raw_results.get('analysis', {})
        content = raw_results.get('content', {})

        # Calculer les FPS car l'analyseur ne les renvoie pas toujours directement
        duration = info.get('duration_seconds', 0)
        frames = info.get('total_frames', 0)
        fps_calc = int(frames / duration) if duration > 0 else 24

        # 3. REFORMATER pour le Frontend (Aplatir les données)
        # C'est ici que l'affichage est réparé
        frontend_data = {
            # -- Métriques --
            "duration_seconds": round(duration, 1),
            "resolution": "HD", # Valeur par défaut
            "fps": fps_calc,
            "face_detection": analysis.get('face_detection', 'Non').replace('_', ' ').title(),
            
            # -- Analyse --
            "has_speech": len(content.get('original_transcription', '')) > 5,
            "topic": analysis.get('main_topic', 'Général').title(),
            "topic_confidence": analysis.get('topic_confidence', 'N/A'),
            
            # -- Sentiment --
            # Le JS cherche 'emotion', pas 'text_emotion'
            "emotion": analysis.get('text_emotion', 'Neutre').title(),
            "emotion_confidence": "80%", # Valeur simulée pour la barre de progression
            
            # -- Transcription --
            "transcribed_text": content.get('original_transcription', 'Pas de transcription.')
        }

        # 4. Nettoyage du fichier vidéo
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass # Ignorer si Windows verrouille le fichier temporairement

        return frontend_data

    except Exception as e:
        print(f"❌ Erreur Task: {e}")
        return {'error': str(e)}