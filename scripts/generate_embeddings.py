import os
import json
import numpy as np
import faiss  # Ajout de FAISS pour l'indexation
from sentence_transformers import SentenceTransformer

# Définition des chemins
INPUT_FILE = "output/extracted_texts.json"
OUTPUT_DIR = "output/"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss_index.bin")

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
all_texts = []
all_embeddings = []

for filename, text in extracted_texts.items():
    print(f"🔍 Génération d'embeddings pour {filename}...")
    sentences = text.split("\n")  # Découper en phrases
    embeddings = model.encode(sentences, convert_to_numpy=True)
    all_texts.extend(sentences)
    all_embeddings.append(embeddings)

# Conversion en tableau numpy
all_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

# Vérification que des embeddings ont bien été générés
if all_embeddings.size == 0:
    print("❌ Erreur : Aucun embedding généré, impossible de créer l'index FAISS.")
    exit(1)

# Sauvegarde des embeddings
np.save(EMBEDDINGS_FILE, all_embeddings)
print(f"✅ Embeddings générés et enregistrés dans '{EMBEDDINGS_FILE}'.")

# Création de l'index FAISS
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance L2
index.add(all_embeddings)  # Ajout des embeddings dans FAISS

# Sauvegarde de l'index FAISS
faiss.write_index(index, INDEX_FILE)
print(f"✅ Index FAISS enregistré dans '{INDEX_FILE}'.")
