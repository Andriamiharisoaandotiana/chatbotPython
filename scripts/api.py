from flask import Flask, request, jsonify
import os
import json
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Définition des chemins
INDEX_FILE = "output/faiss_index.bin"
TEXTS_FILE = "output/extracted_texts.json"

# Vérification de l'existence des fichiers
if not os.path.exists(INDEX_FILE) or not os.path.exists(TEXTS_FILE):
    logging.error("❌ Erreur : Fichiers FAISS ou textes manquants.")
    exit(1)

# Chargement des modèles
logging.info("🔄 Chargement du modèle CamemBERT...")
embed_model = SentenceTransformer("camembert-base")

logging.info("🔄 Chargement du modèle T5...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

logging.info("🔄 Chargement de l'index FAISS...")
index = faiss.read_index(INDEX_FILE)

logging.info("🔄 Chargement des textes extraits...")
with open(TEXTS_FILE, "r", encoding="utf-8") as f:
    extracted_texts = json.load(f)

all_texts = [sentence for text in extracted_texts.values() for sentence in text.split("\n")]

def search_faiss(query, top_k=5):
    """Recherche les passages pertinents avec FAISS."""
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)
    return [all_texts[idx] for idx in indices[0] if 0 <= idx < len(all_texts)]

def generate_answer(context, question):
    """Génère une réponse avec le modèle T5."""
    input_text = f"question: {question} contexte: {context}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Endpoint pour interroger le chatbot."""
    data = request.get_json()
    query = data.get("question")

    if not query:
        return jsonify({"error": "La question est obligatoire"}), 400

    logging.info(f"📩 Requête reçue : {query}")
    relevant_texts = search_faiss(query)
    
    if not relevant_texts:
        generated_response = "Désolé, je n'ai pas trouvé d'information pertinente."
    else:
        context = " ".join(relevant_texts)
        generated_response = generate_answer(context, query)

    logging.info(f"✅ Réponse générée : {generated_response}")
    return jsonify({"message": generated_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
