# ultra_simple_video_analyzer_local.py

import os
import sys
import subprocess
import importlib
import warnings
import json
import tempfile

# ------------------------------
# Step 1: Safe imports
# ------------------------------
def safe_import(module_name, install_name=None):
    if install_name is None:
        install_name = module_name
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} imported successfully")
        return module
    except ImportError:
        print(f"📦 Installing {install_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
        return importlib.import_module(module_name)

# Core libraries
np = safe_import("numpy")
cv2 = safe_import("cv2", "opencv-python")
torch = safe_import("torch")
transformers = safe_import("transformers")
sr = safe_import("speech_recognition")
moviepy_editor = safe_import("moviepy.editor", "moviepy")
spacy = safe_import("spacy")
sumy = safe_import("sumy")
librosa = safe_import("librosa")
PIL = safe_import("PIL", "Pillow")
matplotlib = safe_import("matplotlib")
facenet_pytorch = safe_import("facenet_pytorch")
deep_translator = safe_import("deep_translator")
arabic_reshaper = safe_import("arabic_reshaper")
bidi = safe_import("bidi")
ipd = safe_import("IPython")

warnings.filterwarnings("ignore")
print("✅ All libraries imported successfully!")

# ------------------------------
# Step 2: Define UltraSimpleVideoAnalyzer
# ------------------------------
class UltraSimpleVideoAnalyzer:
    def __init__(self, video_dir="C:/Users/layla/Videos"):
        self.VIDEO_DIR = video_dir
        self.setup_ultra_simple_models()
    
    def setup_ultra_simple_models(self):
        print("🔄 Setting up ultra-simple models...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = sr.Recognizer()
        self.emotion_keywords = {
            'happy': ['happy','joy','excited','good','great','wonderful','amazing'],
            'sad': ['sad','unhappy','cry','bad','terrible','awful','depressed'],
            'angry': ['angry','mad','frustrated','annoyed','upset'],
            'neutral': ['ok','fine','normal','regular','usual']
        }
        self.topic_keywords = {
            'sports':['sport','game','player','team','match','goal','win'],
            'politics':['government','president','minister','election','policy'],
            'entertainment':['movie','music','show','celebrity','film','song'],
            'technology':['computer','phone','internet','software','digital'],
            'culture':['culture','tradition','heritage','art','history'],
            'war':['war','battle','military','soldier','attack']
        }
        print("✅ Ultra-simple models ready!")

    def format_arabic_text(self, text):
        try:
            return bidi.algorithm.get_display(arabic_reshaper.reshape(text))
        except:
            return text

    def get_video_paths(self):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_paths = []
        if os.path.exists(self.VIDEO_DIR):
            for root, dirs, files in os.walk(self.VIDEO_DIR):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_paths.append(os.path.join(root, file))
        return video_paths

    def list_available_videos(self):
        video_paths = self.get_video_paths()
        if not video_paths:
            print(f"📁 No video files found in {self.VIDEO_DIR}")
            return []
        print(f"🎬 Found {len(video_paths)} video files:")
        for i, path in enumerate(video_paths):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"   {i+1}. {os.path.basename(path)} ({size_mb:.1f} MB)")
        return video_paths

    def extract_audio_ultra_simple(self, video_path):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_path = temp_audio.name
            video = moviepy_editor.VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            return audio_path
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return None

    def transcribe_audio_ultra_simple(self, audio_path):
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            languages = ['ar-SA','fr-FR','en-US']
            for lang in languages:
                try:
                    text = self.recognizer.recognize_google(audio, language=lang)
                    if len(text.strip())>2:
                        return text, {'ar-SA':'arabic','fr-FR':'french','en-US':'english'}[lang]
                except:
                    continue
            try:
                text = self.recognizer.recognize_google(audio)
                return text,"auto-detected"
            except:
                return "No speech detected","unknown"
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return "","unknown"

    def detect_language_simple(self,text):
        if len(text.strip())<3:
            return "unknown"
        arabic_chars=set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        if any(c in arabic_chars for c in text):
            return "arabic"
        french_chars=set('éèêëàâäîïôöùûüç')
        if any(c in french_chars for c in text.lower()):
            return "french"
        return "english"

    def translate_text_simple(self,text,source_lang):
        if source_lang=='english' or len(text.strip())<2:
            return text
        lang_map={'arabic':'ar','french':'fr'}
        try:
            return deep_translator.GoogleTranslator(source=lang_map.get(source_lang,'auto'),target='en').translate(text[:500])
        except:
            return text

    def analyze_emotions_simple(self,text):
        text_lower=text.lower()
        scores={e:sum(1 for k in kws if k in text_lower) for e,kws in self.emotion_keywords.items()}
        return max(scores,key=scores.get) if any(s>0 for s in scores.values()) else "neutral"

    def detect_faces_simple(self,video_path):
        cap=cv2.VideoCapture(video_path)
        face_detected=False
        for _ in range(5):
            ret,frame=cap.read()
            if ret:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=self.face_cascade.detectMultiScale(gray,1.1,4)
                if len(faces)>0:
                    face_detected=True
                    break
        cap.release()
        return "faces_detected" if face_detected else "no_faces"

    def classify_topic_simple(self,text):
        text_lower=text.lower()
        scores={t:sum(1 for k in kws if k in text_lower) for t,kws in self.topic_keywords.items()}
        if any(v>0 for v in scores.values()):
            main=max(scores,key=scores.get)
            conf=min(scores[main]/len(text.split())*10,1.0)
            return main,conf
        return "general",0.3

    def summarize_text_simple(self,text):
        if len(text)<30:
            return "Insufficient text for summary"
        sentences=text.split('.')
        if len(sentences)>2:
            return sentences[0]+'. '+sentences[-1]+'.'
        return text[:200]+"..."

    def analyze_video_ultra_simple(self,video_path=None):
        if video_path is None:
            paths=self.get_video_paths()
            if not paths:
                return {"error":"No video files found"}
            video_path=paths[0]
        print(f"🎬 Analyzing: {os.path.basename(video_path)}")
        cap=cv2.VideoCapture(video_path)
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=cap.get(cv2.CAP_PROP_FPS)
        duration=total_frames/fps if fps>0 else 0
        cap.release()
        audio_path=self.extract_audio_ultra_simple(video_path)
        if not audio_path:
            return {"error":"Failed to extract audio"}
        text,lang=self.transcribe_audio_ultra_simple(audio_path)
        final_lang=self.detect_language_simple(text)
        if final_lang!="english":
            text_en=self.translate_text_simple(text,final_lang)
        else:
            text_en=text
        emotion=self.analyze_emotions_simple(text_en)
        face_status=self.detect_faces_simple(video_path)
        topic,conf=self.classify_topic_simple(text_en)
        summary=self.summarize_text_simple(text_en)
        try: os.unlink(audio_path)
        except: pass
        return {
            "video_info":{"file_name":os.path.basename(video_path),"duration_seconds":duration,"total_frames":total_frames,"detected_language":final_lang,"transcribed_text_length":len(text)},
            "analysis":{"face_detection":face_status,"text_emotion":emotion,"main_topic":topic,"topic_confidence":f"{conf:.1%}","video_summary":summary},
            "content":{"original_transcription":text[:350]+"..." if len(text)>350 else text,"english_translation":text_en[:350]+"..." if len(text_en)>350 else text_en}
        }

    def display_ultra_simple_results(self,results):
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        print("\n"+"="*60)
        print("🎯 ULTRA-SIMPLE VIDEO ANALYSIS RESULTS")
        print("="*60)
        vi=results['video_info']
        an=results['analysis']
        ct=results['content']
        print(f"\n📁 VIDEO INFORMATION:")
        print(f"   File: {vi['file_name']}")
        print(f"   Duration: {vi['duration_seconds']:.1f}s")
        print(f"   Language: {vi['detected_language']}")
        print(f"   Transcription Length: {vi['transcribed_text_length']} chars")
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Face Detection: {an['face_detection']}")
        print(f"   Emotion: {an['text_emotion']}")
        print(f"   Main Topic: {an['main_topic']}")
        print(f"   Confidence: {an['topic_confidence']}")
        print(f"\n📄 VIDEO SUMMARY:")
        if vi['detected_language']=="arabic":
            print(f"   {self.format_arabic_text(an['video_summary'])}")
        else:
            print(f"   {an['video_summary']}")
        print(f"\n🗣️ CONTENT PREVIEW:")
        if vi['detected_language']=="arabic":
            print(f"   Original: {self.format_arabic_text(ct['original_transcription'])}")
        else:
            print(f"   Original: {ct['original_transcription']}")
        if ct['english_translation']!=ct['original_transcription']:
            print(f"   English: {ct['english_translation']}")

# ------------------------------
# Step 3: Run analysis
# ------------------------------
def run_ultra_simple_analysis(video_dir="C:/Users/layla/Videos"):
    print("🚀 Starting Ultra-Simple Video Analysis")
    print("="*50)
    analyzer = UltraSimpleVideoAnalyzer(video_dir)
    paths=analyzer.list_available_videos()
    if not paths: return
    results=analyzer.analyze_video_ultra_simple()
    analyzer.display_ultra_simple_results(results)
    # Save results
    try:
        with open("ultra_simple_video_analysis.json","w",encoding="utf-8") as f:
            json.dump(results,f,ensure_ascii=False,indent=2)
        print("\n💾 Results saved to ultra_simple_video_analysis.json")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")
    return results

# ------------------------------
# Step 4: Main
# ------------------------------
if __name__=="__main__":
    run_ultra_simple_analysis()
