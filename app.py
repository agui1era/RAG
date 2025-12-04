from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
from pathlib import Path
import tiktoken

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de Flask y CORS
app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS abierto para el widget; se puede ajustar a dominios específicos si hace falta
CORS(app, resources={r"/chat": {"origins": "*"}})

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables de entorno
API_KEY = os.getenv('OPENAI_API_KEY', 'tu_api_key_default')  # Clave de OpenAI
BASE_DIR = Path(__file__).resolve().parent
PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', './pdfs')         # Directorio de PDFs
PDF_DIRECTORY = Path(PDF_DIRECTORY)
if not PDF_DIRECTORY.is_absolute():
    PDF_DIRECTORY = BASE_DIR / PDF_DIRECTORY
MAX_PAGES = int(os.getenv('MAX_PAGES', 10))                  # Máximo de páginas por PDF
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.0))
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', "Eres un asistente útil de documentos PDF para la compañía XYZ. Tienes acceso a documentos internos. Usa los siguientes fragmentos de contexto para responder la pregunta al final. Si la respuesta no está en el contexto, di que no lo sabes, no intentes inventar una respuesta.")
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 300))
PORT = int(os.getenv('PORT', 9994))
WIDGET_API_KEY = os.getenv('WIDGET_API_KEY', '')
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

import json

# Variables globales para FAISS y QA
docsearch = None
qa = None
PROCESSED_FILES_LOG = BASE_DIR / "processed_files.json"
documents_initialized = False


def load_processed_files():
    """Carga la lista de archivos ya procesados."""
    if PROCESSED_FILES_LOG.exists():
        try:
            with open(PROCESSED_FILES_LOG, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logger.warning(f"{PROCESSED_FILES_LOG} is empty or corrupted. Starting with empty set.")
            return set()
    return set()


def save_processed_files(processed_files):
    """Guarda la lista de archivos procesados."""
    with open(PROCESSED_FILES_LOG, 'w') as f:
        json.dump(list(processed_files), f)


def load_pdfs_from_local(directory_path, processed_files):
    """Carga y procesa solo los PDFs nuevos desde un directorio local."""
    directory_path = Path(directory_path)
    new_text = ""
    new_files = []
    
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory_path.resolve()}")
        return "", []

    for filepath in directory_path.glob("*.pdf"):
        filename = filepath.name
        if filename not in processed_files:
            try:
                with open(filepath, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    pdf_text = ""
                    for i, page in enumerate(pdf_reader.pages[:MAX_PAGES]):
                        pdf_text += page.extract_text()
                    new_text += pdf_text
                    new_text += f"\n\nEnd of document: {filename}\n\n"
                    logger.info(f"Loaded new PDF: {filename}, Pages: {min(MAX_PAGES, len(pdf_reader.pages))}")
                    new_files.append(filename)
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")
                
    return new_text, new_files


def count_tokens(text):
    """Cuenta los tokens en un texto usando tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def embed_documents(docs, existing_index=None):
    """Crea embeddings de los documentos y los guarda/actualiza en FAISS."""
    try:
        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        doc_objs = [Document(page_content=doc) for doc in docs]
        
        if existing_index:
            existing_index.add_documents(doc_objs)
            faiss_index = existing_index
            logger.info("Added new documents to existing FAISS index")
        else:
            faiss_index = FAISS.from_documents(doc_objs, embeddings)
        logger.info("Created new FAISS index")
            
        faiss_index.save_local(str(FAISS_INDEX_DIR))
        logger.info(f"FAISS index saved locally at '{FAISS_INDEX_DIR}'")
        return faiss_index
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise


def initialize_documents():
    """Inicializa documentos desde FAISS y agrega nuevos si existen."""
    global docsearch, qa
    
    processed_files = load_processed_files()
    
    # 1. Intentar cargar el índice existente SOLO si hay archivos procesados
    if processed_files:
        try:
            docsearch = FAISS.load_local(str(FAISS_INDEX_DIR), OpenAIEmbeddings(api_key=API_KEY), allow_dangerous_deserialization=True)
            logger.info(f"FAISS index loaded from local storage at {FAISS_INDEX_DIR}")
        except Exception:
            docsearch = None
            logger.info("No existing FAISS index found or error loading it.")
    else:
        docsearch = None
        logger.info("Processed files list is empty. Starting with a fresh index.")

    # 2. Buscar archivos nuevos
    logger.info(f"Checking for new files in {PDF_DIRECTORY.resolve()}...")
    logger.info(f"Already processed files: {processed_files}")
    new_pdf_text, new_files = load_pdfs_from_local(PDF_DIRECTORY, processed_files)
    
    if new_files:
        logger.info(f"Found {len(new_files)} new files to process.")
        total_tokens = count_tokens(new_pdf_text)
        logger.info(f"Total tokens in new files: {total_tokens}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_text(new_pdf_text)
        
        # 3. Actualizar o crear índice
        docsearch = embed_documents(docs, existing_index=docsearch)
        
        # 4. Actualizar registro de archivos procesados
        processed_files.update(new_files)
        save_processed_files(processed_files)
    else:
        logger.info("No new files found to process.")
        
    if docsearch is None:
        # Si no había índice y no hubo archivos nuevos, crear uno dummy para no fallar
        logger.warning("Index is empty. Creating placeholder.")
        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        docsearch = FAISS.from_documents([Document(page_content="Placeholder")], embeddings)

    # Inicializar el modelo de lenguaje y el sistema de preguntas y respuestas
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate

    llm = ChatOpenAI(api_key=API_KEY, model_name=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    
    # Definir el template del prompt para restringir respuestas usando la variable de entorno
    template = f"""{{system_prompt}}

Context:
{{context}}

Question: {{question}}
Helpful Answer:"""
    
    # Inyectamos el SYSTEM_PROMPT en el template
    final_template = template.format(system_prompt=SYSTEM_PROMPT, context="{context}", question="{question}")
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=final_template)

    RETRIEVER_K = int(os.getenv('RETRIEVER_K', 3))
    
    # Usamos ConversationalRetrievalChain para manejar historial
    # combine_docs_chain_kwargs allows passing the prompt to the internal stuff chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=docsearch.as_retriever(search_kwargs={"k": RETRIEVER_K}),
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    logger.info("Document initialization complete")


@app.before_request
def ensure_documents_initialized():
    """Garantiza que el índice esté listo aunque la app se importe sin ejecutar este módulo directamente."""
    global documents_initialized
    if documents_initialized:
        return
    try:
        initialize_documents()
        documents_initialized = True
    except Exception as e:
        logger.critical(f"Failed to initialize documents on first request: {e}")


@app.route('/')
def index():
    """Sirve la página de prueba."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado del servidor."""
    return jsonify(status='ok'), 200


@app.route('/chat', methods=['GET', 'POST', 'OPTIONS', 'HEAD'])
def chat():
    """Procesa las consultas enviadas al servidor."""
    if request.method in ('OPTIONS', 'HEAD'):
        # Responder rápido a preflight CORS
        return ('', 204)
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Use POST with JSON {prompt, history} to chat.'
        })

    data = request.json or {}
    prompt = data.get('prompt', '')
    history_raw = data.get('history', [])

    # Autorización básica por API key si se configuró
    if WIDGET_API_KEY:
        provided_key = request.headers.get('x-api-key')
        if provided_key != WIDGET_API_KEY:
            logger.warning("Unauthorized request: invalid API key")
            return jsonify({'error': 'Unauthorized'}), 401

    if not prompt:
        logger.warning("Request missing required 'prompt' field")
        return jsonify({'error': 'Missing required field: prompt'}), 400

    # Convertir historial al formato esperado por LangChain [(q, a), (q, a)]
    chat_history = []
    temp_q = None
    for msg in history_raw:
        if msg.get('role') == 'user':
            temp_q = msg.get('content')
        elif msg.get('role') == 'assistant' and temp_q:
            chat_history.append((temp_q, msg.get('content')))
            temp_q = None

    try:
        logger.info(f"Processing prompt: {prompt[:50]}...")
        logger.info(f"History length: {len(chat_history)}")
        
        # ConversationalRetrievalChain espera 'question' y 'chat_history'
        result = qa({"question": prompt, "chat_history": chat_history})
        response_text = result['answer']
        
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


@app.after_request
def add_cors_headers(response):
    """Asegura que las respuestas incluyan encabezados CORS adecuados."""
    response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
    response.headers.add('Vary', 'Origin')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, x-api-key')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, HEAD')
    return response


if __name__ == '__main__':
    try:
        initialize_documents()
        documents_initialized = True
    except Exception as e:
        logger.critical(f"Failed to initialize documents: {e}")
        # No salir, permitir que arranque para servir estáticos aunque falle la IA
    app.run(debug=True, host='0.0.0.0', port=PORT)
