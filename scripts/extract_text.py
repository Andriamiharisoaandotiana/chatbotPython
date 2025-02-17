import pdfplumber
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_DIR = "data/"  # Dossier contenant les fichiers PDF
OUTPUT_DIR = "output/"  # Dossier de sortie
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "extracted_texts.json")

# Vérification et création des dossiers si nécessaires
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extrait et nettoie le texte d'un fichier PDF"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        if not text.strip():
            logging.warning(f"⚠️ Aucun texte détecté dans {pdf_path}.")
            return None

        # Nettoyage : suppression des lignes vides
        cleaned_text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])

        logging.info(f"🔍 Aperçu du texte extrait ({os.path.basename(pdf_path)[:30]}...): {cleaned_text[:500]}...")

        return cleaned_text
    except Exception as e:
        logging.error(f"❌ Erreur lors de l'extraction de {pdf_path}: {e}")
        return None

def extract_all_pdfs():
    """Parcourt tous les fichiers PDF du dossier et extrait le texte"""
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logging.warning("⚠️ Aucun fichier PDF trouvé dans 'data/'. Ajoutez des fichiers et réessayez.")
        return None

    extracted_texts = {}

    # Utilisation de ThreadPoolExecutor pour l'extraction parallèle
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda file: (file, extract_text_from_pdf(os.path.join(DATA_DIR, file))), pdf_files)

    for file, extracted_text in results:
        if extracted_text:
            extracted_texts[file] = extracted_text
            logging.info(f"✅ Extraction réussie : {file}")
        else:
            logging.warning(f"⚠️ Aucune donnée utile extraite de {file}.")

    return extracted_texts if extracted_texts else None

def save_extracted_texts(texts):
    """Enregistre les textes extraits dans un fichier JSON"""
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Texte enregistré dans '{OUTPUT_FILE}'")
    except Exception as e:
        logging.error(f"❌ Erreur lors de l'enregistrement du fichier JSON: {e}")

if __name__ == "__main__":
    texts = extract_all_pdfs()
    if texts:
        save_extracted_texts(texts)
    else:
        logging.info("❌ Aucune donnée à sauvegarder.")
