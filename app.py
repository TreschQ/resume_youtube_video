import streamlit as st
import os
from pytube import YouTube
from moviepy.editor import AudioFileClip
import whisper
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import json
import unidecode
import requests
import time

# Charger les variables d'environnement
load_dotenv()

# Configuration des APIs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Mode debug
debug_mode = st.sidebar.checkbox("Mode Debug", value=False)

# Sélection du modèle
model_option = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Groq (Mixtral-8x7b)", "OpenAI (gpt-3.5-turbo)"]
)

def debug_log(message):
    if debug_mode:
        st.write(f"DEBUG - {message}")

# Vérification des clés API
if model_option.startswith("Groq"):
    if not GROQ_API_KEY:
        st.error("⚠️ La clé API Groq n'est pas configurée. Veuillez vérifier votre fichier .env")
        st.stop()
    if not GROQ_API_KEY.startswith("gsk_"):
        st.error("⚠️ Format de clé API Groq invalide. La clé doit commencer par 'gsk_'")
        st.stop()
elif model_option.startswith("OpenAI"):
    if not OPENAI_API_KEY:
        st.error("⚠️ La clé API OpenAI n'est pas configurée. Veuillez vérifier votre fichier .env")
        st.stop()

def call_ai_api(messages):
    if model_option.startswith("Groq"):
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        api_url = GROQ_API_URL
        model_name = "mixtral-8x7b-32768"
    else:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        api_url = OPENAI_API_URL
        model_name = "gpt-3.5-turbo"
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        debug_log(f"Envoi de la requête à {model_option}")
        debug_log(f"Headers: {headers}")
        debug_log(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            api_url,
            headers=headers,
            json=data
        )
        debug_log(f"Status code: {response.status_code}")
        
        if response.status_code == 401:
            service = "Groq" if model_option.startswith("Groq") else "OpenAI"
            st.error(f"🔑 Erreur d'authentification avec l'API {service}. Veuillez vérifier votre clé API.")
            if service == "Groq":
                st.info("Pour obtenir une nouvelle clé API Groq :")
                st.markdown("""
                1. Rendez-vous sur [console.groq.com](https://console.groq.com)
                2. Connectez-vous ou créez un compte
                3. Allez dans la section "API Keys"
                4. Créez une nouvelle clé
                5. Copiez la clé dans votre fichier .env
                """)
            else:
                st.info("Pour obtenir une nouvelle clé API OpenAI :")
                st.markdown("""
                1. Rendez-vous sur [platform.openai.com](https://platform.openai.com)
                2. Connectez-vous ou créez un compte
                3. Allez dans la section "API Keys"
                4. Créez une nouvelle clé
                5. Copiez la clé dans votre fichier .env
                """)
            raise Exception(f"Clé API {service} invalide")
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            debug_log(f"Response body: {response.text}")
            raise Exception(f"Erreur API: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        debug_log(f"Erreur de requête: {str(e)}")
        raise Exception(f"Erreur de connexion à l'API: {str(e)}")

def get_youtube_id(url):
    try:
        yt = YouTube(url)
        return yt.video_id
    except Exception as e:
        raise Exception(f"URL YouTube invalide : {str(e)}")

def get_youtube_subtitles(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Essayer d'abord en français
        try:
            transcript = transcript_list.find_transcript(['fr'])
        except:
            # Si pas de français, prendre l'original et traduire
            try:
                transcript = transcript_list.find_manually_created_transcript()
            except:
                try:
                    # Si pas de sous-titres manuels, prendre les auto-générés
                    transcript = transcript_list.find_generated_transcript()
                except:
                    st.warning("Aucun sous-titre trouvé pour cette vidéo")
                    return None
        
        # Traduire en français si nécessaire
        try:
            if transcript.language_code != 'fr':
                transcript = transcript.translate('fr')
        except Exception as e:
            st.warning(f"Impossible de traduire les sous-titres en français : {str(e)}")
        
        # Récupérer le texte complet
        full_transcript = ' '.join([str(entry['text']).strip() for entry in transcript.fetch()])
        return full_transcript
    except Exception as e:
        st.error(f"Erreur lors de la récupération des sous-titres : {str(e)}")
        return None

def download_youtube_audio(url, output_path="temp", max_retries=3):
    try:
        # Créer le dossier temporaire s'il n'existe pas
        os.makedirs(output_path, exist_ok=True)
        
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                debug_log(f"Tentative de téléchargement {retry_count + 1}/{max_retries}")
                
                # Configuration de pytube avec des en-têtes personnalisés
                yt = YouTube(
                    url,
                    use_oauth=False,
                    allow_oauth_cache=True
                )
                
                # Configuration des en-têtes
                yt.bypass_age_gate()
                
                # Attendre que les informations de la vidéo soient chargées
                debug_log("Récupération des informations de la vidéo...")
                try:
                    title = yt.title
                    debug_log(f"Vidéo trouvée : {title}")
                except Exception as e:
                    debug_log(f"Erreur lors de la récupération du titre : {str(e)}")
                    debug_log("Tentative de continuer malgré l'erreur...")
                
                # Récupérer tous les flux disponibles
                debug_log("Recherche des flux audio disponibles...")
                streams = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc()
                
                if not streams:
                    raise Exception("Aucun flux audio disponible")
                
                # Sélectionner le meilleur flux audio
                audio_stream = streams.first()
                if not audio_stream:
                    raise Exception("Impossible de trouver un flux audio approprié")
                
                debug_log(f"Flux audio sélectionné : {audio_stream.abr if hasattr(audio_stream, 'abr') else 'Qualité inconnue'}")
                
                # Télécharger en MP4
                debug_log("Téléchargement du fichier audio...")
                mp4_path = audio_stream.download(output_path)
                
                if not os.path.exists(mp4_path):
                    raise Exception("Le fichier audio n'a pas été téléchargé correctement")
                
                debug_log("Conversion en MP3...")
                # Convertir en MP3
                mp3_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(mp4_path))[0]}.mp3")
                
                try:
                    audio_clip = AudioFileClip(mp4_path)
                    audio_clip.write_audiofile(mp3_path, verbose=False, logger=None)
                    audio_clip.close()
                except Exception as e:
                    debug_log(f"Erreur lors de la conversion en MP3 : {str(e)}")
                    if os.path.exists(mp4_path):
                        os.remove(mp4_path)
                    raise e
                
                # Nettoyer
                if os.path.exists(mp4_path):
                    os.remove(mp4_path)
                
                if not os.path.exists(mp3_path):
                    raise Exception("La conversion en MP3 a échoué")
                
                debug_log("Téléchargement et conversion terminés avec succès")
                return mp3_path
                
            except Exception as e:
                last_error = str(e)
                debug_log(f"Erreur lors de la tentative {retry_count + 1}: {last_error}")
                retry_count += 1
                if retry_count < max_retries:
                    debug_log("Nouvelle tentative dans 2 secondes...")
                    time.sleep(2)
                    
        raise Exception(f"Échec après {max_retries} tentatives. Dernière erreur : {last_error}")
        
    except Exception as e:
        debug_log(f"Erreur fatale lors du téléchargement : {str(e)}")
        raise Exception(f"Erreur lors du téléchargement : {str(e)}")

def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        raise Exception(f"Erreur lors de la transcription : {str(e)}")

def generate_summary(text):
    try:
        # Log initial
        debug_log(f"Longueur du texte d'entrée: {len(text)}")
        
        # Convertir immédiatement tout en ASCII
        text = unidecode.unidecode(str(text).strip())
        
        # Créer le prompt avec un texte sans accents
        messages = [
            {
                "role": "system",
                "content": "Tu es un assistant specialise dans la creation de resumes clairs et concis."
            },
            {
                "role": "user",
                "content": "Voici un texte a resumer. Fais-en un resume structure avec les points cles :\n\n" + text
            }
        ]
        
        debug_log("Texte converti en ASCII")
        
        try:
            # Appel à l'API sélectionnée
            summary = call_ai_api(messages)
            debug_log(f"Appel API {model_option} réussi")
            debug_log(f"Réponse reçue, longueur: {len(summary)}")
            return summary
            
        except Exception as e:
            debug_log(f"Erreur API: {type(e).__name__} - {str(e)}")
            raise e
            
    except Exception as e:
        debug_log(f"Erreur générale: {type(e).__name__} - {str(e)}")
        raise Exception(f"Erreur lors de la génération du résumé : {str(e)}")

# Configuration de Whisper
model = whisper.load_model("base")

# Interface Streamlit
st.title("📝 YouTube Transcripteur & Résumeur")
st.write("Collez un lien YouTube pour obtenir sa transcription et un résumé")

# Input pour l'URL YouTube
youtube_url = st.text_input("URL de la vidéo YouTube")

if st.button("Analyser"):
    if youtube_url:
        try:
            # Récupérer l'ID de la vidéo
            video_id = get_youtube_id(youtube_url)
            
            # Essayer d'abord de récupérer les sous-titres
            transcript = get_youtube_subtitles(video_id)
            
            if transcript:
                st.success("✅ Sous-titres récupérés directement depuis YouTube!")
            else:
                st.info("⚠️ Pas de sous-titres disponibles, utilisation de la transcription audio...")
                with st.spinner("Téléchargement de l'audio..."):
                    audio_path = download_youtube_audio(youtube_url)
                
                with st.spinner("Transcription en cours..."):
                    transcript = transcribe_audio(audio_path)
                    os.remove(audio_path)
            
            # st.subheader("Transcription")
            # st.write(transcript)
            
            with st.spinner("Génération du résumé..."):
                summary = generate_summary(transcript)
                st.subheader("Résumé")
                st.write(summary)
            
        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")
    else:
        st.warning("Veuillez entrer une URL YouTube valide")