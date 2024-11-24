# Wolof ASR Training with OpenAI Whisper

Ce dépôt contient le code, la documentation et les modèles générés pour entraîner un système de **Reconnaissance Automatique de la Parole (ASR)** en wolof, en utilisant le modèle **OpenAI Whisper** et les données du dataset [Wolof TTS](https://huggingface.co/datasets/galsenai/wolof_tts).

## 🚀 Objectif du projet

L'objectif de ce projet est de créer un modèle performant pour la reconnaissance vocale automatique (ASR) en wolof, une langue africaine largement parlée au Sénégal. Ce projet vise à :
- Développer une **solution robuste** pour la transcription audio-texte en wolof.
- Contribuer à la préservation et à la valorisation des langues africaines à travers des outils technologiques.
- Faciliter l'intégration des systèmes ASR dans des applications locales.

---

## 📚 Dataset

Le projet utilise le dataset public suivant :
- **[Wolof TTS](https://huggingface.co/datasets/galsenai/wolof_tts)** : un corpus contenant des données audios et leurs transcriptions en wolof.

### Préparation des données
Les données sont automatiquement téléchargées et traitées pour être utilisées dans le pipeline d'entraînement. Le script convertit le format des fichiers pour qu'ils soient compatibles avec le modèle **Whisper**.

---

## 🏗️ Structure du projet

Voici la structure principale du dépôt :

```
.
├── data/                          # Dossier contenant les données d'entrée et pré-traitées
├── src/                           # Code source
│   ├── dataset_loader.py          # Script pour charger et préparer le dataset
│   ├── training.py                # Script d'entraînement du modèle
│   ├── evaluation.py              # Script d'évaluation
│   ├── utils.py                   # Fonctions utilitaires
├── models/                        # Dossier pour stocker les modèles générés
│   ├── experiment_0/
│   │   ├── checkpoints/           # Sauvegardes des epochs
│   │   └── final_model.pt         # Modèle final entraîné
├── README.md                      # Documentation du projet
├── requirements.txt               # Dépendances Python
└── config.yaml                    # Fichier de configuration du projet
```

---

## 🛠️ Entraînement

### Étapes pour exécuter le projet :

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/votre-utilisateur/wolof-asr.git
   cd wolof-asr
   ```

2. **Installer les dépendances** :
   Créez un environnement virtuel (optionnel mais recommandé) :
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Configurer les paramètres** :
   Modifiez le fichier `config.yaml` pour ajuster les hyperparamètres tels que :
   - Nombre d'époques
   - Taille du batch
   - Learning rate

4. **Lancer l'entraînement** :
   Exécutez le script d'entraînement :
   ```bash
   python app.py
   ```

5. **Suivi en temps réel avec Weights & Biases** :
   Connectez-vous à votre compte **Weights & Biases** pour surveiller les métriques.

---

## 📊 Évaluation

Après l'entraînement, évaluez le modèle sur un dataset de test pour calculer des métriques comme :
- **WER (Word Error Rate)** : mesure de précision de la transcription.
- **CER (Character Error Rate)** : pour des transcriptions plus granulaires.

Exécutez :
```bash
python src/evaluation.py --model_path=models/experiment_0/final_model.pt
```

---

## 📂 Modèles générés

### Modèles disponibles
| Modèle            | Taille     | Architecture            | Lien                                                                                  |
|--------------------|------------|--------------------------|---------------------------------------------------------------------------------------|
| Whisper-Wolof-Small | ~240M      | `openai/whisper-small`   | [Télécharger](https://huggingface.co/)                   |
| Whisper-Wolof-Tiny | ~75M       | `openai/whisper-tiny`    | [Télécharger](https://huggingface.co/)                    |

---

## 🙌 Contributeurs

Nous remercions les contributeurs pour leur travail et leur engagement dans ce projet.

- **[Mamadou Diagne](https://github.com/dofbi)** - Développeur principal
- **[Nom du contributeur](https://github.com/contributeur)** - Support technique
- **[GalsenAI](https://huggingface.co/galsenai)** - Fournisseur du dataset

Si vous souhaitez contribuer, merci de soumettre une *Pull Request* ou de contacter [votre email].

---

## 📜 Licence

Ce projet est sous licence **MIT**. Vous êtes libre d'utiliser, de modifier et de distribuer le code, sous réserve de mentionner les auteurs originaux.

---

## 🌍 Impact attendu

Ce projet constitue une avancée technologique majeure pour la numérisation des langues africaines, avec des applications possibles dans l'éducation, les technologies vocales et la préservation culturelle.
