from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import tiktoken

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de Flask y CORS
app = Flask(__name__)
CORS(app)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables de entorno
API_KEY = os.getenv('OPENAI_API_KEY', 'tu_api_key_default')  # Clave de OpenAI
PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', './pdfs')         # Directorio de PDFs
MAX_PAGES = int(os.getenv('MAX_PAGES', 10))                  # Máximo de páginas por PDF

# Variables globales para FAISS y QA
docsearch = None
qa = None


def load_pdfs_from_local(directory_path):
    """Carga y procesa los PDFs desde un directorio local."""
    all_text = ""
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'rb') as f:
                pdf_reader = PdfReader(f)
                pdf_text = ""
                for i, page in enumerate(pdf_reader.pages[:MAX_PAGES]):
                    pdf_text += page.extract_text()
                all_text += pdf_text
                all_text += f"\n\nEnd of document: {filename}\n\n"
                logger.info(f"Loaded PDF: {filename}, Pages: {min(MAX_PAGES, len(pdf_reader.pages))}")
    return all_text


def count_tokens(text):
    """Cuenta los tokens en un texto usando tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def embed_documents(docs):
    """Crea embeddings de los documentos y los guarda en FAISS."""
    try:
        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        doc_objs = [Document(page_content=doc) for doc in docs]
        faiss_index = FAISS.from_documents(doc_objs, embeddings)
        faiss_index.save_local("faiss_index")
        logger.info("FAISS index saved locally at 'faiss_index'")
        return faiss_index
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise


def initialize_documents():
    """Inicializa documentos desde FAISS o crea un nuevo índice."""
    global docsearch, qa
    try:
        # Intentar cargar el índice FAISS existente
        docsearch = FAISS.load_local("faiss_index", OpenAIEmbeddings(api_key=API_KEY))
        logger.info("FAISS index loaded from local storage")
    except Exception:
        logger.info("No FAISS index found. Processing documents...")
        # Procesar documentos si no se encuentra el índice
        all_pdf_text = load_pdfs_from_local(PDF_DIRECTORY)
        total_tokens_before_split = count_tokens(all_pdf_text)
        logger.info(f"Total tokens before splitting: {total_tokens_before_split}")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_text(all_pdf_text)
        docsearch = embed_documents(docs)

    # Inicializar el modelo de lenguaje y el sistema de preguntas y respuestas
    from langchain.llms import OpenAI
    llm = OpenAI(api_key=API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    logger.info("Document initialization complete")


@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado del servidor."""
    return jsonify(status='ok'), 200


@app.route('/chat', methods=['POST'])
def chat():
    """Procesa las consultas enviadas al servidor."""
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        logger.warning("Request missing required 'prompt' field")
        return jsonify({'error': 'Missing required field: prompt'}), 400

    try:
        logger.info(f"Processing prompt: {prompt[:50]}...")
        response_text = qa.run(prompt)
        logger.info("Request successful")
        response = {
            "choices": [
                {
                    "message": {
                        "content": response_text
                    }
                }
            ]
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    try:
        initialize_documents()
    except Exception as e:
        logger.critical(f"Failed to initialize documents: {e}")
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=8080)