# recherche et génération des réponses
import os
import json
import numpy as np
import faiss
import sys
import logging
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Définition des chemins
INDEX_FILE = "output/faiss_index.bin"
EMBEDDINGS_FILE = "output/embeddings.npy"
TEXTS_FILE = "output/extracted_texts.json"

# Vérification de l'existence des fichiers
for file in [INDEX_FILE, EMBEDDINGS_FILE, TEXTS_FILE]:
    if not os.path.exists(file):
        logging.error(f"❌ Erreur : Le fichier '{file}' est manquant.")
        exit(1)

# Chargement du modèle CamemBERT pour encoder les requêtes
logging.info("🔄 Chargement du modèle CamemBERT...")
embed_model = SentenceTransformer("camembert-base")

# Chargement du modèle T5 pour générer les réponses
logging.info("🔄 Chargement du modèle T5...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")  # Utilisation d'un modèle plus puissant
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Chargement de l'index FAISS
logging.info("🔄 Chargement de l'index FAISS...")
index = faiss.read_index(INDEX_FILE)

# Chargement des textes extraits
logging.info("🔄 Chargement des textes extraits...")
with open(TEXTS_FILE, "r", encoding="utf-8") as f:
    extracted_texts = json.load(f)

# Transformation en liste de textes
all_texts = [sentence for text in extracted_texts.values() for sentence in text.split("\n")]

# Vérification de la correspondance entre FAISS et les textes
if len(all_texts) != index.ntotal:
    logging.warning(f"⚠️ Le nombre de textes ({len(all_texts)}) et d'embeddings dans FAISS ({index.ntotal}) ne correspond pas.")

def search_faiss(query, top_k=5):
    """Recherche les passages les plus pertinents avec FAISS."""
    logging.info(f"🔎 Recherche FAISS pour la requête : {query}")

    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:  # Indices des passages trouvés
        if 0 <= idx < len(all_texts):
            results.append(all_texts[idx])

    logging.info(f"📄 Passages trouvés : {results}")
    return results

def generate_answer(context, question):
    """Génère une réponse avec le modèle T5."""
    logging.info("🤖 Génération de réponse avec T5...")
    
    if not context:
        return "Désolé, je n'ai pas trouvé d'information pertinente."

    input_text = f"question: {question} contexte: {context}"
    
    # Journaliser l'entrée envoyée à T5
    logging.info(f"📜 Texte envoyé à T5 : {input_text[:500]}")  # Limité à 500 caractères pour éviter un trop gros log

    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = t5_model.generate(
        **inputs,
        max_length=150,  # Augmentation de la longueur max
        num_beams=8,     # Plus de faisceaux pour une meilleure qualité
        temperature=0.7,  # Augmente la diversité des réponses
        top_p=0.9,       # Nucleus sampling pour plus de variété
        early_stopping=True
    )
    
    generated_response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"✅ Réponse générée : {generated_response}")
    return generated_response

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Mode API (appelé depuis Spring Boot)
        query = sys.argv[1]
        logging.info(f"📩 Requête reçue : {query}")

        relevant_texts = search_faiss(query)

        # Sélectionner un meilleur contexte en prenant les 3 meilleurs passages
        context = " ".join(relevant_texts[:3]) if relevant_texts else ""

        generated_response = generate_answer(context, query)

        response = {"message": generated_response}

        # 🔥 Retourne un JSON propre pour Spring Boot
        sys.stdout.reconfigure(encoding='utf-8')
        print(json.dumps(response, ensure_ascii=False))
        sys.stdout.flush()

    else:
        # Mode interactif (pour test local)
        print("\n🤖 Chatbot Loi de Finances - Posez une question (tapez 'exit' pour quitter)\n")
        while True:
            query = input("Vous : ")
            if query.lower() == "exit":
                print("👋 Fin de la session.")
                break

            relevant_texts = search_faiss(query)
            context = " ".join(relevant_texts[:3]) if relevant_texts else ""

            response = generate_answer(context, query)
            print(f"🤖 Chatbot : {response}")
