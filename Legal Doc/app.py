from flask import Flask, render_template, request, jsonify, send_file
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import fitz  # PyMuPDF for PDF handling
import docx
from deep_translator import GoogleTranslator
import os
import pytesseract  # OCR for scanned PDFs
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import io
from fpdf import FPDF
import re  # Import regex for cleaning text

app = Flask(__name__, static_folder="static", template_folder="templates")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load AI models
summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
classification_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ✅ NEW: FLAN-T5 Model for Chatbot & Summary Refinement (No Login Required)
flan_model = "google/flan-t5-large"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model)

SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "fr": "French", "es": "Spanish", "ta": "Tamil",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "te": "Telugu", "ur": "Urdu", "pa": "Punjabi"
}

def clean_text(text):
    """Fixes OCR errors and cleans extracted text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?()'’\-\s]", "", text)
    text = text.replace("’", "'")
    return text.strip()

def extract_text(file_path):
    """Extracts text from PDFs, DOCX, TXT, and scanned PDFs using OCR."""
    text = ""

    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():  
                text += page_text + "\n\n"
            else:  
                img = page.get_pixmap()
                img_bytes = img.tobytes("jpeg")
                img_pil = Image.open(io.BytesIO(img_bytes))
                text += pytesseract.image_to_string(img_pil, lang="eng") + "\n\n"

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    return clean_text(text)

def refine_text_with_flan(text):
    """Uses FLAN-T5 to improve the readability of the summary."""
    input_text = f"Refine this legal text for clarity and readability:\n{text}"
    input_tokens = flan_tokenizer.encode(input_text, return_tensors="pt")
    output_tokens = flan_model.generate(input_tokens, max_length=500)
    return flan_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

def chunk_text(text, chunk_size=500):
    """Splits long text into smaller chunks for summarization."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze uploaded legal documents."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    text = extract_text(file_path)

    if not text.strip():
        return jsonify({"error": "Could not extract text from the document."}), 400

    summaries = []
    for chunk in chunk_text(text, chunk_size=500):
        try:
            summary = summary_pipeline(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            refined_summary = refine_text_with_flan(summary)  # ✅ FLAN-T5 improves clarity
            summaries.append(refined_summary)
        except:
            summaries.append("Error summarizing this section.")
    
    formatted_summary = "\n\n".join(summaries)

    try:
        outputs = classification_pipeline(text[:1024], candidate_labels=["Property Dispute", "Criminal Case", "Corporate Law", "Civil Case", "Other Legal Matters"])
        classification = outputs['labels'][0]
    except:
        classification = "Classification error."

    risk_score = "Low"
    if "dispute" in classification.lower() or "criminal" in classification.lower():
        risk_score = "High"
    elif "corporate" in classification.lower():
        risk_score = "Medium"

    return jsonify({"summary": formatted_summary, "classification": classification, "risk_score": risk_score})

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Uses FLAN-T5 to answer user questions about the document."""
    data = request.json
    question = data.get("question", "").strip()
    document_text = data.get("text", "")

    if not question:
        return jsonify({"answer": "Please enter a valid question."})

    input_text = f"Document Context: {document_text}\n\nQuestion: {question}\nAnswer:"
    input_tokens = flan_tokenizer.encode(input_text, return_tensors="pt")
    output_tokens = flan_model.generate(input_tokens, max_length=200)
    answer = flan_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
