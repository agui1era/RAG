from flask import Flask, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import PyPDF2
import os

app = Flask(__name__)

# Configuración global
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
vectorstore = None

# Función para leer texto de un PDF desde una ruta
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Procesar el PDF al iniciar el servidor
@app.before_first_request
def process_pdf():
    global vectorstore
    pdf_path = "document.pdf"  # Nombre del archivo en la ruta del script
    if not os.path.exists(pdf_path):
        print(f"Error: No se encontró el archivo {pdf_path} en la ruta actual.")
        return

    # Extraer texto del PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text.strip():
        print("Error: El PDF no contiene texto legible.")
        return

    # Dividir el texto en párrafos y crear documentos
    paragraphs = [para.strip() for para in pdf_text.split("\n") if para.strip()]
    documents = [Document(page_content=para) for para in paragraphs]

    # Crear el vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"Embeddings generados con éxito para {len(paragraphs)} párrafos.")

# Endpoint para realizar consultas al PDF procesado
@app.route('/query', methods=['POST'])
def query_pdf():
    global vectorstore
    if not vectorstore:
        return jsonify({"error": "El PDF no se ha procesado correctamente."}), 400

    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No se proporcionó una consulta."}), 400

    # Generar embeddings para la consulta
    results = vectorstore.similarity_search(query, k=3)
    return jsonify({
        "results": [
            {"content": result.page_content, "score": result.metadata.get('score', None)}
            for result in results
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)