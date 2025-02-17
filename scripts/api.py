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
logging.info("🔄 Chargement du modèle CamemBERT...")
embed_model = SentenceTransformer("camembert-base")

logging.info("🔄 Chargement du modèle T5...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-large")

logging.info("🔄 Chargement du modèle de re-ranking...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

logging.info("🔄 Chargement de l'index FAISS...")
index = faiss.read_index(INDEX_FILE)

logging.info("🔄 Chargement des textes extraits...")
with open(TEXTS_FILE, "r", encoding="utf-8") as f:
    extracted_texts = json.load(f)

# Mise en place d'une liste de tous les textes
all_texts = [sentence for text in extracted_texts.values() for sentence in text.split("\n")] if extracted_texts else []

def search_faiss(query, top_k=10):
    """Recherche les passages pertinents avec FAISS et applique un re-ranking."""
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [all_texts[idx] for idx in indices[0] if 0 <= idx < len(all_texts)]

    # Supprimer les passages courts (< 10 caractères) et éliminer les doublons
    filtered_results = list(set([r for r in results if len(r) > 10]))

    if not filtered_results:
        return []

    # Re-ranking des passages retournés par FAISS
    ranking_scores = reranker.predict([(query, passage) for passage in filtered_results])
    ranked_results = [text for _, text in sorted(zip(ranking_scores, filtered_results), reverse=True)]

    logging.info(f"📄 Passages pertinents après re-ranking : {ranked_results[:5]}")
    return ranked_results[:5]  # On garde les 5 meilleurs

def generate_answer(context, question):
    """Génère une réponse avec le modèle T5."""
    if not context:
        return "Désolé, je n'ai pas trouvé d'information pertinente."

    input_text = f"question: {question} contexte: {context}"
    logging.info(f"📜 Texte envoyé à T5 : {input_text[:500]}")

    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = t5_model.generate(
        **inputs,
        max_length=150,  # Limite la longueur pour éviter les répétitions
        num_beams=5,  # Réduit les choix pour éviter des réponses aléatoires
        temperature=0.7,  # Rend la réponse plus naturelle
        do_sample=True,  # Active l'échantillonnage
        top_p=0.9,  # Sélectionne les tokens les plus probables
        early_stopping=True
    )

    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Vérification si la réponse est identique à la question (ce qui est un problème)
    if response.lower() == question.lower():
        logging.warning("⚠️ La réponse générée est identique à la question. Ajustement en cours...")
        return "Je ne peux pas fournir une réponse précise pour cette question."

    logging.info(f"✅ Réponse générée : {response}")
    return response


    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Vérification si la réponse est pertinente
    if len(response.split()) < 5:
        response = "Je ne peux pas fournir une réponse précise pour cette question."

    logging.info(f"✅ Réponse générée : {response}")
    return response

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
        context = " ".join(relevant_texts[:3])  # Limiter le contexte à 3 passages pertinents
        generated_response = generate_answer(context, query)

    logging.info(f"✅ Réponse générée : {generated_response}")
    return jsonify({"message": generated_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
