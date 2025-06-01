import streamlit as st
import os
from yt_dlp import YoutubeDL
from openai import OpenAI
from groq import Groq
from pydub import AudioSegment
import math
import google.generativeai as genai

# --- Configuration de la Page Streamlit (DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT) ---
st.set_page_config(layout="wide")

# --- Configuration des clés API ---
# Il est recommandé d'utiliser les secrets Streamlit ou les variables d'environnement pour les clés API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Initialisation et vérification des clients API ---

# Mettre les messages liés aux API dans la sidebar après set_page_config
# et seulement si l'initialisation pose problème ou pour confirmation.

def initialize_apis():
    """Initialise les clients API et affiche les erreurs/succès dans la sidebar."""
    global client_openai, client_groq # Indiquer qu'on modifie les variables globales
    
    openai_status = False
    groq_status = False
    gemini_status = False

    # OpenAI
    if not OPENAI_API_KEY:
        st.sidebar.warning("Clé API OpenAI non configurée. Le résumé OpenAI ne sera pas disponible.")
    else:
        try:
            client_openai = OpenAI(api_key=OPENAI_API_KEY)
            st.sidebar.info("Client OpenAI initialisé (non utilisé par défaut pour le résumé).")
            openai_status = True
        except Exception as e:
            st.sidebar.error(f"Erreur init OpenAI: {e}")

    # Groq
    if not GROQ_API_KEY:
        st.sidebar.error("Clé API Groq NON configurée. Transcription impossible.")
    else:
        try:
            client_groq = Groq(api_key=GROQ_API_KEY)
            st.sidebar.success("API Groq configurée pour la transcription.")
            groq_status = True
        except Exception as e:
            st.sidebar.error(f"Erreur init Groq: {e}")
            client_groq = None # S'assurer qu'il est None en cas d'échec

    # Gemini
    if not GOOGLE_API_KEY:
        st.sidebar.error("Clé API Google Gemini NON configurée. Résumé Gemini impossible.")
    else:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            st.sidebar.success("API Google Gemini configurée pour le résumé.")
            gemini_status = True
        except Exception as e:
            st.sidebar.error(f"Erreur configuration Gemini: {e}")
    
    # Stopper l'application si les API critiques (Groq pour transcription, Gemini pour résumé) ne sont pas prêtes
    if not groq_status:
        st.error("L'API Groq est requise pour la transcription et n'a pas pu être initialisée. Vérifiez votre clé GROQ_API_KEY.")
        st.stop()
    if not gemini_status:
        st.error("L'API Google Gemini est requise pour le résumé et n'a pas pu être initialisée. Vérifiez votre clé GOOGLE_API_KEY.")
        st.stop()

# Initialiser les clients API à None au début
client_openai = None
client_groq = None
# genai est configuré globalement, pas besoin d'un client stocké de la même manière pour l'instant.

initialize_apis() # Appeler la fonction d'initialisation

# --- Constantes ---
MAX_FILE_SIZE_MB_GROQ = 20  # Limite de sécurité pour la taille des fichiers envoyés à Groq (en Mo)
CHUNK_OVERLAP_MS = 5000     # Chevauchement de 5 secondes entre les chunks pour une meilleure continuité

# --- Fonctions ---

def download_audio(youtube_url, output_filename="audio_extrait"):
    """Télécharge l'audio d'une vidéo YouTube au format MP3."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_filename}.%(ext)s',
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            st.info(f"Début du téléchargement de l'audio depuis : {youtube_url}")
            error_code = ydl.download([youtube_url])
            if error_code == 0:
                mp3_filename = f"{output_filename}.mp3"
                st.success("Téléchargement audio terminé avec succès.")
                return mp3_filename
            else:
                st.error(f"Erreur lors du téléchargement audio (code: {error_code}).")
                return None
    except Exception as e:
        st.error(f"Exception lors du téléchargement : {e}")
        return None

def transcribe_audio_chunk_groq(audio_chunk_path, chunk_number):
    """Transcrire un unique morceau d'audio avec Groq."""
    st.info(f"Transcription du morceau {chunk_number} avec Groq...")
    try:
        with open(audio_chunk_path, "rb") as file_data:
            transcription = client_groq.audio.transcriptions.create(
                file=(os.path.basename(audio_chunk_path), file_data.read()),
                model="whisper-large-v3",
            )
        return transcription.text
    except Exception as e:
        st.error(f"Erreur lors de la transcription du morceau {chunk_number} avec Groq : {e}")
        return None
    finally:
        if os.path.exists(audio_chunk_path):
            try:
                os.remove(audio_chunk_path)
            except Exception as e_del:
                st.warning(f"Impossible de supprimer le fichier audio temporaire {audio_chunk_path}: {e_del}")

def transcribe_audio_groq(audio_path):
    """Transcrire l'audio en utilisant l'API Groq avec Whisper, en gérant le découpage (chunking)."""
    if not os.path.exists(audio_path):
        st.error(f"Le fichier audio {audio_path} n'a pas été trouvé.")
        return None

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    st.info(f"Taille du fichier audio : {file_size_mb:.2f} Mo.")

    full_transcript = ""

    if file_size_mb <= MAX_FILE_SIZE_MB_GROQ:
        st.info("Le fichier est suffisamment petit, transcription en une seule fois.")
        transcript_text = transcribe_audio_chunk_groq(audio_path, 1)
        if transcript_text:
            full_transcript = transcript_text
    else:
        st.info(f"Le fichier dépasse {MAX_FILE_SIZE_MB_GROQ} Mo. Découpage en morceaux nécessaire...")
        try:
            audio = AudioSegment.from_mp3(audio_path)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier audio avec Pydub : {e}. Assurez-vous que FFmpeg est correctement installé et accessible.")
            return None
        
        chunk_length_ms = 10 * 60 * 1000  # 10 minutes en millisecondes
        
        num_chunks = math.ceil(len(audio) / (chunk_length_ms - CHUNK_OVERLAP_MS if chunk_length_ms > CHUNK_OVERLAP_MS else chunk_length_ms))
        if num_chunks == 0 and len(audio) > 0:
             num_chunks = 1

        st.info(f"Découpage en {num_chunks} morceau(x) de ~10 minutes avec chevauchement de {CHUNK_OVERLAP_MS / 1000}s.")

        transcripts_list = []
        all_chunks_processed_successfully = True

        for i in range(num_chunks):
            start_ms = max(0, i * (chunk_length_ms - CHUNK_OVERLAP_MS))
            end_ms = min(len(audio), start_ms + chunk_length_ms)
            
            chunk = audio[start_ms:end_ms]
            chunk_filename = f"temp_chunk_{i+1}.mp3"
            
            st.info(f"Exportation du morceau {i+1}/{num_chunks} ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s)...")
            try:
                chunk.export(chunk_filename, format="mp3")
            except Exception as e_export:
                st.error(f"Erreur lors de l'exportation du morceau {chunk_filename}: {e_export}")
                all_chunks_processed_successfully = False
                break

            chunk_size_mb = os.path.getsize(chunk_filename) / (1024 * 1024)
            if chunk_size_mb > MAX_FILE_SIZE_MB_GROQ + 5:
                 st.warning(f"Le morceau {chunk_filename} ({chunk_size_mb:.2f} Mo) est plus grand que prévu. Tentative de transcription quand même...")
            
            transcript_part = transcribe_audio_chunk_groq(chunk_filename, i + 1)
            
            if transcript_part:
                transcripts_list.append(transcript_part)
            else:
                st.error(f"La transcription du morceau {i+1} a échoué.")
                all_chunks_processed_successfully = False

        if all_chunks_processed_successfully and transcripts_list:
            full_transcript = " ".join(transcripts_list)
            st.success("Tous les morceaux ont été transcrits et combinés.")
        elif transcripts_list:
            full_transcript = " ".join(transcripts_list)
            st.warning("Certains morceaux n'ont pas pu être transcrits. Le résultat peut être incomplet.")
        else:
            st.error("La transcription de tous les morceaux a échoué.")
            return None
            
    if full_transcript:
        st.success("Transcription globale terminée.")
        return full_transcript
    else:
        st.error("La transcription a échoué ou n'a produit aucun texte.")
        return None

def summarize_text_gemini(text_to_summarize, model_name="gemini-1.5-flash-latest"):
    """Résume le texte en utilisant l'API Google Gemini."""
    if not GOOGLE_API_KEY:
        st.error("La clé API Google Gemini n'est pas configurée pour le résumé.")
        return None
    st.info(f"Démarrage du résumé avec Google Gemini ({model_name})...")
    prompt = f"""
    Voici la transcription d'une vidéo YouTube. Fais-en un résumé concis et détaillé en français. Extrait les points clés, les arguments principaux, les conclusions et toutes les informations importantes. Le résumé doit être bien structuré et facile à lire.

    Transcription :
    --- DEBUT DE LA TRANSCRIPTION ---
    {text_to_summarize}
    --- FIN DE LA TRANSCRIPTION ---

    Résumé détaillé en français :
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # Vérifier si la réponse contient du texte avant d'y accéder
        if response.parts:
            summary = response.text
        else:
            # Tenter de récupérer le texte via une autre méthode ou logguer l'erreur
            summary = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            if not summary:
                 st.warning(f"La réponse de Gemini ne semble pas contenir de texte directement. Contenu de la réponse : {response}")
                 # Essayer de voir si c'est une erreur de blocage
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                     st.error(f"Le prompt a été bloqué par Gemini. Raison : {response.prompt_feedback.block_reason}")
                     if response.prompt_feedback.safety_ratings:
                         for rating in response.prompt_feedback.safety_ratings:
                             st.error(f"  - Catégorie: {rating.category}, Probabilité: {rating.probability}")
                     return "Résumé bloqué par le filtre de contenu de Gemini."
                 else:
                     return "Impossible d'extraire le résumé de la réponse Gemini (pas de texte direct ou erreur de blocage non spécifiée)."

        st.success(f"Résumé avec Gemini ({model_name}) terminé.")
        return summary
    except Exception as e:
        st.error(f"Erreur lors du résumé avec Gemini ({model_name}): {e}")
        # Afficher plus de détails sur l'erreur si possible
        if hasattr(e, 'response') and e.response:
            st.error(f"Détails de la réponse de l'erreur Gemini: {e.response.text}")
        return None

# Fonction OpenAI conservée pour référence ou fallback éventuel, mais non utilisée dans le flux principal.
def summarize_text_openai(text_to_summarize):
    """Résume le texte en utilisant l'API OpenAI (GPT-3.5 Turbo)."""
    if not client_openai:
        st.error("Le client OpenAI n'est pas initialisé (clé API manquante ?). Résumé OpenAI impossible.")
        return None
    st.info("Démarrage du résumé avec OpenAI (GPT-3.5 Turbo)...")
    prompt = f"""
    Voici le transcript d'une vidéo. Veuillez en faire un résumé concis en français, en extrayant les points clés et les informations les plus importantes.

    Transcript :
    {text_to_summarize}

    Résumé concis en français :
    """
    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant IA expert en résumé de texte en français."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content
        st.success("Résumé avec OpenAI terminé.")
        return summary
    except Exception as e:
        st.error(f"Erreur lors du résumé avec OpenAI : {e}")
        return None

# --- Interface Streamlit ---
st.title("Résumé de Vidéo YouTube  YT -> 📜 -> 📝")
st.markdown("Entrez l'URL d'une vidéo YouTube pour obtenir sa transcription (Groq Whisper) et un résumé (Google Gemini).")

youtube_url = st.text_input("URL de la vidéo YouTube :", placeholder="https://www.youtube.com/watch?v=...")

if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None


if st.button("Obtenir le résumé 🚀", type="primary"):
    if youtube_url:
        st.session_state.transcription = None
        st.session_state.summary = None
        st.session_state.audio_file_path = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Préparation..."):
            status_text.info("Nettoyage des anciens fichiers temporaires...")
            for f in os.listdir("."):
                if f.startswith("temp_chunk_") and f.endswith(".mp3") or f == "audio_extrait.mp3":
                    try: os.remove(f) 
                    except: pass
        
        with st.spinner("Traitement en cours... Veuillez patienter. Cela peut prendre plusieurs minutes."):
            status_text.info("Étape 1/3: Téléchargement de l'audio...")
            progress_bar.progress(10)
            
            main_audio_file = download_audio(youtube_url, output_filename="audio_extrait")
            st.session_state.audio_file_path = main_audio_file
            progress_bar.progress(33)

            if main_audio_file:
                status_text.info("Étape 2/3: Transcription de l'audio (peut être long)...")
                transcript = transcribe_audio_groq(main_audio_file)
                st.session_state.transcription = transcript
                progress_bar.progress(66)

                if transcript:
                    status_text.info("Étape 3/3: Résumé du texte avec Gemini...")
                    # Utilisation de Gemini pour le résumé
                    summary_text = summarize_text_gemini(transcript)
                    st.session_state.summary = summary_text
                    progress_bar.progress(100)
                    if summary_text:
                        status_text.success("Traitement terminé !")
                    else:
                        status_text.error("Le résumé avec Gemini a échoué.")
                else:
                    st.warning("La transcription a échoué, le résumé ne peut pas être généré.")
                    status_text.warning("La transcription a échoué.")
            else:
                st.warning("Le téléchargement audio a échoué, impossible de continuer.")
                status_text.warning("Le téléchargement audio a échoué.")
        
        if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
            try:
                os.remove(st.session_state.audio_file_path)
                st.info(f"Fichier audio principal {st.session_state.audio_file_path} supprimé.")
            except Exception as e:
                st.warning(f"Impossible de supprimer le fichier audio principal {st.session_state.audio_file_path}: {e}")
        
        status_text.info("Nettoyage final des fichiers temporaires...")
        for f in os.listdir("."):
            if f.startswith("temp_chunk_") and f.endswith(".mp3"):
                try: os.remove(f)
                except: pass
            
    else:
        st.warning("Veuillez entrer une URL YouTube.")

if st.session_state.transcription:
    st.subheader("📜 Transcription de la vidéo (par Groq Whisper) :")
    st.text_area("Texte transcrit", st.session_state.transcription, height=300)

if st.session_state.summary:
    st.subheader("📝 Résumé de la vidéo (par Google Gemini) :")
    st.markdown(st.session_state.summary)

st.sidebar.header("À propos")
st.sidebar.info(
    "Cette application utilise :\n"
    "- `yt-dlp` pour télécharger l'audio.\n"
    "- `Groq` (Whisper-large-v3) pour la transcription.\n"
    "- `Google Gemini` (gemini-1.5-flash-latest) pour le résumé."
)
st.sidebar.markdown("---")
st.sidebar.markdown("N'oubliez pas de configurer vos clés API `GROQ_API_KEY` et `GOOGLE_API_KEY` comme variables d'environnement.")
if client_openai:
    st.sidebar.markdown("La clé `OPENAI_API_KEY` est aussi détectée si vous souhaitez modifier le code pour utiliser OpenAI pour le résumé.") 