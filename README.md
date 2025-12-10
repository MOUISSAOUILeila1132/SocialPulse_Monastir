# 🎬 Video Analyzer AI (Local Engine)

![Video Analyzer AI Banner](https://via.placeholder.com/1000x300/2a0a45/ffffff?text=Video+Analyzer+AI)

**Video Analyzer AI** est un outil puissant et autonome d'analyse vidéo locale. Ce script Python combine la vision par ordinateur, le traitement du langage naturel (NLP) et la reconnaissance vocale pour extraire des informations détaillées de vos fichiers vidéo.

> *Basé sur le concept de la plateforme "Video Analyzer AI", cette version locale exécute l'analyse directement sur votre machine sans envoyer de données dans le cloud.*

## ✨ Fonctionnalités Principales

Le script `ultra_simple_video_analyzer_local.py` intègre plusieurs technologies d'IA pour fournir une analyse complète :

*   **🗣️ Transcription Vocale (Speech-to-Text)** : Extraction et transcription de l'audio via Google Speech Recognition.
*   **🌍 Détection & Traduction de Langue** : Supporte l'anglais, le français et l'arabe (avec gestion de l'affichage bidirectionnel pour l'arabe). Traduction automatique vers l'anglais pour l'analyse.
*   **🎭 Analyse de Sentiments** : Détecte les émotions dominantes dans le discours (Joyeux, Triste, Colère, Neutre).
*   **👤 Détection Faciale** : Utilise OpenCV pour vérifier la présence de visages humains dans la vidéo.
*   **🏷️ Classification de Sujet** : Catégorise le contenu (Sport, Politique, Tech, Guerre, Culture, etc.).
*   **📝 Résumé Automatique** : Génère un résumé concis du contenu vidéo.
*   **💾 Export JSON** : Sauvegarde automatiquement toutes les métadonnées et analyses dans un fichier structuré.

## 🛠️ Prérequis

*   **Python 3.8+** installé sur votre machine.
*   Une connexion internet (requise pour l'installation automatique des paquets et les API de traduction/reconnaissance vocale).

## 🚀 Installation

Ce projet a été conçu pour être **ultra-simple** à installer. Le script gère lui-même ses dépendances.

1.  Clonez ce dépôt ou téléchargez le fichier `ultra_simple_video_analyzer_local.py`.
2.  Assurez-vous d'avoir Python installé.
3.  C'est tout ! Le script installera automatiquement les librairies manquantes (`numpy`, `opencv`, `torch`, `transformers`, etc.) lors de la première exécution.

## ⚙️ Configuration

Par défaut, le script cherche les vidéos dans le dossier `C:/Users/layla/Videos`.

**Pour analyser vos propres vidéos :**

1.  Ouvrez le fichier `ultra_simple_video_analyzer_local.py` avec un éditeur de texte (Notepad, VS Code, etc.).
2.  Cherchez la ligne **48** (dans la classe `UltraSimpleVideoAnalyzer`) :
    ```python
    def __init__(self, video_dir="C:/Users/layla/Videos"):
    ```
3.  Remplacez le chemin par celui de votre dossier vidéo, par exemple :
    ```python
    def __init__(self, video_dir="./mes_videos"):
    ```

## ▶️ Utilisation

Ouvrez un terminal (invite de commande) dans le dossier du projet et lancez :

```bash
python ultra_simple_video_analyzer_local.py
