# YouTube Transcripteur & Résumeur

Cette application Streamlit permet de :
1. Télécharger l'audio d'une vidéo YouTube
2. Transcrire l'audio en texte
3. Générer un résumé intelligent du contenu

## Prérequis

- Python 3.8+
- Une clé API Groq

## Installation

1. Cloner le dépôt :
```bash
git clone [url-du-repo]
cd [nom-du-repo]
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer la clé API :
- Renommer le fichier `.env.example` en `.env`
- Remplacer `votre_clé_api_ici` par votre clé API Groq

## Utilisation

1. Lancer l'application :
```bash
streamlit run app.py
```

2. Ouvrir votre navigateur à l'adresse indiquée (généralement http://localhost:8501)
3. Coller l'URL d'une vidéo YouTube
4. Cliquer sur "Analyser"

## Notes

- L'application crée un dossier temporaire `temp` pour stocker les fichiers audio
- Les fichiers sont automatiquement supprimés après traitement
- La transcription utilise le modèle "base" de Whisper
- Le résumé est généré avec le modèle Mixtral 8x7B via Groq 