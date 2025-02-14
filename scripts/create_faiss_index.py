#stockage FAISS
import os
import faiss
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Définition des chemins
EMBEDDINGS_FILE = "output/embeddings.npy"
INDEX_FILE = "output/faiss_index.bin"
OUTPUT_DIR = "output/"

# Vérification et création du dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vérification de l'existence du fichier d'embeddings
if not os.path.exists(EMBEDDINGS_FILE):
    logging.error(f"❌ Erreur : Le fichier '{EMBEDDINGS_FILE}' n'existe pas. Exécutez d'abord 'generate_embeddings.py'.")
    exit(1)

# Chargement des embeddings
try:
    embeddings = np.load(EMBEDDINGS_FILE)
    if embeddings.size == 0:
        logging.error("❌ Erreur : Le fichier d'embeddings est vide.")
        exit(1)
except Exception as e:
    logging.error(f"❌ Erreur lors du chargement des embeddings : {e}")
    exit(1)

# Vérification des dimensions des embeddings
if len(embeddings.shape) != 2:
    logging.error("❌ Erreur : Les embeddings ne sont pas dans un format correct (2D array attendu).")
    exit(1)

embedding_dim = embeddings.shape[1]
logging.info(f"📊 Dimension des embeddings : {embedding_dim}")

# Création d'un index FAISS optimisé
logging.info("🔄 Création de l'index FAISS...")

# Méthode standard (IndexFlatL2)
# index = faiss.IndexFlatL2(embedding_dim)  # Recherche exacte basée sur la distance euclidienne

# Méthode optimisée avec HNSW pour une recherche plus rapide
index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 voisins dans le graphe

# Ajout des embeddings à l'index
index.add(embeddings)
logging.info(f"✅ {index.ntotal} embeddings ajoutés à l'index FAISS.")

# Sauvegarde de l'index
faiss.write_index(index, INDEX_FILE)
logging.info(f"💾 Index FAISS sauvegardé dans '{INDEX_FILE}'.")
