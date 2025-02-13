#génératin des embeddings
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Définition des chemins
INPUT_FILE = "output/extracted_texts.json"
OUTPUT_DIR = "output/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")

# Vérification et création du dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vérification que le fichier extrait existe
if not os.path.exists(INPUT_FILE):
    print(f"❌ Erreur : Le fichier '{INPUT_FILE}' n'existe pas. Exécutez d'abord 'extract_text.py'.")
    exit(1)

# Chargement des textes extraits
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        extracted_texts = json.load(f)
        if not extracted_texts:
            print("❌ Erreur : Aucun texte extrait trouvé dans le fichier JSON.")
            exit(1)
except json.JSONDecodeError:
    print("❌ Erreur : Impossible de lire 'extracted_texts.json'. Vérifiez son contenu.")
    exit(1)

# Initialisation du modèle d'embedding
print("🔄 Chargement du modèle CamemBERT...")
model = SentenceTransformer("camembert-base")

# Génération des embeddings
embeddings = {}

for filename, text in extracted_texts.items():
    print(f"🔍 Génération d'embeddings pour {filename}...")
    sentences = text.split("\n")  # Découper en phrases pour de meilleurs résultats
    embeddings[filename] = model.encode(sentences, convert_to_numpy=True)

# Conversion en tableau numpy
all_embeddings = np.array([emb for emb_list in embeddings.values() for emb in emb_list])

# Sauvegarde des embeddings
np.save(OUTPUT_FILE, all_embeddings)
print(f"\n✅ Embeddings générés et enregistrés dans '{OUTPUT_FILE}'.")
