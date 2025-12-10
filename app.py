# app.py
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tasks import celery, analyze_video_task
from celery.result import AsyncResult
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration Flask
app.secret_key = os.getenv("SECRET_KEY", "dev_secret")
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", "uploads")
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_CONTENT_SIZE", 104857600))  # 100 Mo

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Config Celery
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)
celery.conf.update(app.config)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/start-analysis", methods=["POST"])
def start_analysis():
    if "video_file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé."}), 400

    file = request.files["video_file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné."}), 400

    try:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # Lancer la tâche Celery
        task = analyze_video_task.delay(video_path)
        return jsonify({"task_id": task.id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status/<task_id>", methods=["GET"])
def task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'En attente...'}
    elif task.state != 'FAILURE':
        response = {'state': task.state, 'status': 'Analyse en cours...'}
        if task.state == 'SUCCESS':
            response['result'] = task.result
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port)