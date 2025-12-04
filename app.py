from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
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
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables de entorno
API_KEY = os.getenv('OPENAI_API_KEY', 'tu_api_key_default')  # Clave de OpenAI
PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', './pdfs')         # Directorio de PDFs
MAX_PAGES = int(os.getenv('MAX_PAGES', 10))                  # Máximo de páginas por PDF
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 0))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4-turbo')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.0))
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', "You are a helpful assistant. Use the following pieces of context to answer the question at the end. If the answer is not in the context, say that you don't know, don't try to make up an answer.")
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 500))
PORT = int(os.getenv('PORT', 9994))

import json

# Variables globales para FAISS y QA
docsearch = None
qa = None
PROCESSED_FILES_LOG = "processed_files.json"


def load_processed_files():
    """Carga la lista de archivos ya procesados."""
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, 'r') as f:
            return set(json.load(f))
    return set()


def save_processed_files(processed_files):
    """Guarda la lista de archivos procesados."""
    with open(PROCESSED_FILES_LOG, 'w') as f:
        json.dump(list(processed_files), f)


def load_pdfs_from_local(directory_path, processed_files):
    """Carga y procesa solo los PDFs nuevos desde un directorio local."""
    new_text = ""
    new_files = []
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")
        return "", []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf') and filename not in processed_files:
            filepath = os.path.join(directory_path, filename)
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
            
        faiss_index.save_local("faiss_index")
        logger.info("FAISS index saved locally at 'faiss_index'")
        return faiss_index
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise


def initialize_documents():
    """Inicializa documentos desde FAISS y agrega nuevos si existen."""
    global docsearch, qa
    
    processed_files = load_processed_files()
    
    # 1. Intentar cargar el índice existente
    try:
        docsearch = FAISS.load_local("faiss_index", OpenAIEmbeddings(api_key=API_KEY))
        logger.info("FAISS index loaded from local storage")
    except Exception:
        docsearch = None
        logger.info("No existing FAISS index found.")

    # 2. Buscar archivos nuevos
    new_pdf_text, new_files = load_pdfs_from_local(PDF_DIRECTORY, processed_files)
    
    if new_files:
        logger.info(f"Found {len(new_files)} new files to process.")
        total_tokens = count_tokens(new_pdf_text)
        logger.info(f"Total tokens in new files: {total_tokens}")
        
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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

    # Usamos ConversationalRetrievalChain para manejar historial
    # combine_docs_chain_kwargs allows passing the prompt to the internal stuff chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=docsearch.as_retriever(),
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    logger.info("Document initialization complete")


@app.route('/')
def index():
    """Sirve la página de prueba."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado del servidor."""
    return jsonify(status='ok'), 200


@app.route('/chat', methods=['POST'])
def chat():
    """Procesa las consultas enviadas al servidor."""
    data = request.json
    prompt = data.get('prompt', '')
    history_raw = data.get('history', [])

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


if __name__ == '__main__':
    try:
        initialize_documents()
    except Exception as e:
        logger.critical(f"Failed to initialize documents: {e}")
        # No salir, permitir que arranque para servir estáticos aunque falle la IA
    app.run(debug=True, host='0.0.0.0', port=PORT)