import os
import json
import numpy as np
import faiss
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder
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
try:
    logging.info("🔄 Chargement du modèle CamemBERT...")
    embed_model = SentenceTransformer("camembert/camembert-base")

    logging.info("🔄 Chargement du modèle Flan-T5...")
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    logging.info("🔄 Chargement du modèle de re-ranking...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    logging.error(f"❌ Erreur lors du chargement des modèles : {e}")
    exit(1)

# Chargement de l'index FAISS
try:
    logging.info("🔄 Chargement de l'index FAISS...")
    index = faiss.read_index(INDEX_FILE)
except Exception as e:
    logging.error(f"❌ Erreur lors du chargement de l'index FAISS : {e}")
    exit(1)

# Chargement des textes extraits
try:
    logging.info("🔄 Chargement des textes extraits...")
    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        extracted_texts = json.load(f)
    all_texts = [sentence for text in extracted_texts.values() for sentence in text.split("\n")] if extracted_texts else []
except Exception as e:
    logging.error(f"❌ Erreur lors du chargement des textes extraits : {e}")
    exit(1)

def search_faiss(query, top_k=10):
    """Recherche les passages pertinents avec FAISS et applique un re-ranking."""
    try:
        query_embedding = embed_model.encode([query], normalize_embeddings=True)
        distances, indices = index.search(query_embedding, top_k)

        results = [all_texts[idx] for idx in indices[0] if 0 <= idx < len(all_texts)]
        filtered_results = list(set([r for r in results if len(r) > 10]))

        if not filtered_results:
            return []

        ranking_scores = reranker.predict([(query, passage) for passage in filtered_results])
        ranked_results = [text for _, text in sorted(zip(ranking_scores, filtered_results), reverse=True)]
        return ranked_results[:5]
    except Exception as e:
        logging.error(f"❌ Erreur dans la recherche FAISS : {e}")
        return []

def generate_answer(context, question):
    """Génère une réponse avec le modèle T5."""
    if not context:
        return "Désolé, je n'ai pas trouvé d'information pertinente."

    try:
        input_text = f"question: {question} contexte: {context}"
        inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = t5_model.generate(
            **inputs,
            max_length=150,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            early_stopping=True
        )
        response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if response.lower() == question.lower():
            return "Je ne peux pas fournir une réponse précise pour cette question."
        return response
    except Exception as e:
        logging.error(f"❌ Erreur lors de la génération de réponse : {e}")
        return "Une erreur est survenue lors de la génération de la réponse."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Endpoint pour interroger le chatbot."""
    try:
        data = request.get_json()
        query = data.get("question")
        if not query:
            return jsonify({"error": "La question est obligatoire"}), 400

        logging.info(f"📩 Requête reçue : {query}")
        relevant_texts = search_faiss(query)
        
        if not relevant_texts:
            generated_response = "Désolé, je n'ai pas trouvé d'information pertinente."
        else:
            context = " ".join(relevant_texts[:3])
            generated_response = generate_answer(context, query)

        return jsonify({"message": generated_response})
    except Exception as e:
        logging.error(f"❌ Erreur dans l'API : {e}")
        return jsonify({"error": "Une erreur est survenue"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
