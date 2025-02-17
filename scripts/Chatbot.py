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

# Vérification des fichiers
for file in [INDEX_FILE, EMBEDDINGS_FILE, TEXTS_FILE]:
    if not os.path.exists(file):
        logging.error(f"❌ Erreur : Le fichier '{file}' est manquant.")
        exit(1)

# Chargement des modèles
logging.info("🔄 Chargement du modèle CamemBERT...")
embed_model = SentenceTransformer("camembert-base")

logging.info("🔄 Chargement du modèle T5...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Chargement de FAISS et des textes
logging.info("🔄 Chargement de l'index FAISS...")
index = faiss.read_index(INDEX_FILE)

logging.info("🔄 Chargement des textes extraits...")
with open(TEXTS_FILE, "r", encoding="utf-8") as f:
    extracted_texts = json.load(f)
all_texts = [sentence for text in extracted_texts.values() for sentence in text.split("\n")]

if len(all_texts) != index.ntotal:
    logging.warning(f"⚠️ Mismatch entre le nombre de textes ({len(all_texts)}) et FAISS ({index.ntotal})")

def search_faiss(query, top_k=10):
    """Recherche des passages pertinents avec FAISS."""
    logging.info(f"🔎 Recherche FAISS pour : {query}")
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)
    
    results = [all_texts[idx] for idx in indices[0] if 0 <= idx < len(all_texts) and len(all_texts[idx]) > 10]
    logging.info(f"📄 Passages retenus : {results}")
    return results

def refine_context(passages, max_length=512):
    """Filtrer et concaténer les passages les plus pertinents."""
    filtered = [p for p in passages if len(p.split()) > 3]
    context = " ".join(filtered[:5])
    return context[:max_length]

def generate_answer(context, question):
    """Génération de réponse avec T5."""
    logging.info("🤖 Génération avec T5...")
    if not context:
        return "Désolé, je n'ai pas trouvé d'information pertinente."

    input_text = f"question: {question} contexte: {context}"
    logging.info(f"📜 Contexte envoyé à T5 : {input_text[:500]}")
    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = t5_model.generate(
        **inputs, max_length=150, num_beams=8, temperature=0.7, top_p=0.9, early_stopping=True
    )
    
    generated_response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"✅ Réponse : {generated_response}")
    return generated_response

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        logging.info(f"📩 Requête : {query}")
        relevant_texts = search_faiss(query)
        context = refine_context(relevant_texts)
        response = generate_answer(context, query)
        print(json.dumps({"message": response}, ensure_ascii=False))
        sys.stdout.flush()
    else:
        print("\n🤖 Chatbot Loi de Finances - Tapez 'exit' pour quitter\n")
        while True:
            query = input("Vous : ")
            if query.lower() == "exit":
                print("👋 Fin de session.")
                break
            relevant_texts = search_faiss(query)
            context = refine_context(relevant_texts)
            response = generate_answer(context, query)
            print(f"🤖 Chatbot : {response}")
