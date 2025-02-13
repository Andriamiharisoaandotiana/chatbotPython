#Extraction du texte des pdf
import pdfplumber
import os
import json

DATA_DIR = "data/"  # Dossier où se trouvent les PDF
OUTPUT_DIR = "output/"  # Dossier où stocker les textes extraits
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "extracted_texts.json")

# Vérifie si les dossiers existent, sinon les crée
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """ Extrait le texte d'un fichier PDF """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

def extract_all_pdfs():
    """ Parcourt tous les fichiers PDF du dossier et extrait le texte """
    extracted_texts = {}

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ Aucun fichier PDF trouvé dans le dossier 'data/'. Ajoutez des fichiers et réessayez.")
        return None

    for file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, file)
        print(f"📄 Extraction de {file}...")
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if extracted_text:
            extracted_texts[file] = extracted_text
        else:
            print(f"⚠️ Aucun texte détecté dans {file}.")

    return extracted_texts

if __name__ == "__main__":
    texts = extract_all_pdfs()
    
    if texts:
        # Sauvegarde des textes extraits dans un fichier JSON
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ Extraction terminée ! Texte enregistré dans '{OUTPUT_FILE}'")
    else:
        print("❌ Aucune donnée à sauvegarder.")
