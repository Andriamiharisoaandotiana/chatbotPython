#recherche et génération des réponses
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Définition des chemins
INDEX_FILE = "output/faiss_index.bin"
EMBEDDINGS_FILE = "output/embeddings.npy"
TEXTS_FILE = "output/extracted_texts.json"

# Vérification des fichiers
for file in [INDEX_FILE, EMBEDDINGS_FILE, TEXTS_FILE]:
    if not os.path.exists(file):
        print(f"❌ Erreur : Le fichier '{file}' est manquant. Assurez-vous d'avoir exécuté tous les scripts nécessaires.")
        exit(1)

# Chargement du modèle CamemBERT pour l'encodage des requêtes
print("🔄 Chargement du modèle CamemBERT...")
embed_model = SentenceTransformer("camembert-base")

# Chargement du modèle T5 pour la génération de réponses
print("🔄 Chargement du modèle T5...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Chargement de l'index FAISS
print("📥 Chargement de l'index FAISS...")
index = faiss.read_index(INDEX_FILE)

# Chargement des textes extraits pour récupérer les réponses
with open(TEXTS_FILE, "r", encoding="utf-8") as f:
    extracted_texts = json.load(f)

# Transformation du JSON en une liste de textes pour correspondre aux embeddings
all_texts = []
for filename, text in extracted_texts.items():
    all_texts.extend(text.split("\n"))  # Chaque phrase est stockée individuellement

# Vérification de la correspondance entre FAISS et les textes
if len(all_texts) != index.ntotal:
    print("⚠️ Avertissement : Le nombre de textes et d'embeddings dans FAISS ne correspond pas.")

def search_faiss(query, top_k=3):
    """Recherche les passages les plus pertinents avec FAISS."""
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:  # Indices des passages trouvés
        if 0 <= idx < len(all_texts):
            results.append(all_texts[idx])

    return results

def generate_answer(context, question):
    """Génère une réponse avec le modèle T5."""
    input_text = f"question: {question} contexte: {context}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(**inputs, max_length=100)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("\n🤖 Chatbot Loi de Finances - Posez une question (tapez 'exit' pour quitter)\n")
    
    while True:
        query = input("Vous : ")
        if query.lower() == "exit":
            print("👋 Fin de la session.")
            break

        # Recherche des passages pertinents
        relevant_texts = search_faiss(query)

        if not relevant_texts:
            print("🤖 Chatbot : Désolé, je n'ai pas trouvé d'information pertinente.")
        else:
            # Génération de la réponse avec T5
            context = " ".join(relevant_texts)  # Concatène les passages trouvés
            response = generate_answer(context, query)
            print(f"🤖 Chatbot : {response}")
