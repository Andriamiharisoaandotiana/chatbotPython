#stockage FAISS
import os
import faiss
import numpy as np

# Définition des chemins
EMBEDDINGS_FILE = "output/embeddings.npy"
INDEX_FILE = "output/faiss_index.bin"
OUTPUT_DIR = "output/"

# Vérification et création du dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vérification que le fichier embeddings existe
if not os.path.exists(EMBEDDINGS_FILE):
    print(f"❌ Erreur : Le fichier '{EMBEDDINGS_FILE}' n'existe pas. Exécutez d'abord 'generate_embeddings.py'.")
    exit(1)

# Chargement des embeddings
try:
    embeddings = np.load(EMBEDDINGS_FILE)
    if embeddings.size == 0:
        print("❌ Erreur : Le fichier d'embeddings est vide.")
        exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement des embeddings : {e}")
    exit(1)

# Récupération de la dimension des vecteurs
embedding_dim = embeddings.shape[1]
print(f"📊 Dimension des embeddings : {embedding_dim}")

# Création de l'index FAISS
print("🔄 Création de l'index FAISS...")
index = faiss.IndexFlatL2(embedding_dim)  # Index L2 pour la recherche de similarité

# Ajout des embeddings à l'index
index.add(embeddings)
print(f"✅ {index.ntotal} embeddings ajoutés à l'index FAISS.")

# Sauvegarde de l'index
faiss.write_index(index, INDEX_FILE)
print(f"💾 Index FAISS sauvegardé dans '{INDEX_FILE}'.")


