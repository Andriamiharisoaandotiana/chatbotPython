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
    logging.error(f"❌ Erreur : Le fichier '{EMBEDDINGS_FILE}' est introuvable. Exécutez d'abord 'generate_embeddings.py'.")
    exit(1)

# Chargement des embeddings
try:
    embeddings = np.load(EMBEDDINGS_FILE)
    
    if embeddings.size == 0:
        logging.error("❌ Erreur : Le fichier d'embeddings est vide.")
        exit(1)

    # Vérification du format des embeddings
    if not np.issubdtype(embeddings.dtype, np.floating):
        logging.error("❌ Erreur : Les embeddings doivent être de type float32 ou float64.")
        exit(1)

    # Conversion en float32 pour FAISS
    embeddings = embeddings.astype(np.float32)

except Exception as e:
    logging.error(f"❌ Erreur lors du chargement des embeddings : {e}")
    exit(1)

# Vérification des dimensions des embeddings
if len(embeddings.shape) != 2:
    logging.error("❌ Erreur : Les embeddings doivent être un tableau 2D (nb_exemples, dimension).")
    exit(1)

nb_embeddings, embedding_dim = embeddings.shape
logging.info(f"📊 Nombre d'embeddings : {nb_embeddings}, Dimension : {embedding_dim}")

# Création d'un index FAISS optimisé
logging.info("🔄 Création de l'index FAISS...")

try:
    # Vérifier si un index FAISS existe déjà et s'il est valide
    if os.path.exists(INDEX_FILE):
        try:
            existing_index = faiss.read_index(INDEX_FILE)
            if existing_index.d != embedding_dim:
                logging.warning(f"⚠️ Dimension différente dans l'index existant ({existing_index.d} au lieu de {embedding_dim}). Réinitialisation de l'index.")
            else:
                logging.info("🔄 Un index existant a été trouvé et sera mis à jour.")
        except Exception as e:
            logging.warning(f"⚠️ L'ancien fichier FAISS est corrompu ou incompatible. Il sera recréé. Erreur : {e}")

    # Création de l'index FAISS avec HNSW pour une recherche rapide
    index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 voisins dans le graphe

    # Ajout des embeddings à l'index
    index.add(embeddings)
    logging.info(f"✅ {index.ntotal} embeddings ajoutés à l'index FAISS.")

    # Sauvegarde de l'index
    faiss.write_index(index, INDEX_FILE)
    logging.info(f"💾 Index FAISS sauvegardé dans '{INDEX_FILE}'.")

except Exception as e:
    logging.error(f"❌ Erreur lors de la création/sauvegarde de l'index FAISS : {e}")
    exit(1)
