from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from dotenv import load_dotenv
import os
from pathlib import Path
from rag_engine import RAGService

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de Flask y CORS
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/chat": {"origins": "*"}})

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables de entorno y constantes
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
ENV_FIELDS = [
    {"key": "OPENAI_API_KEY", "label": "OpenAI API Key", "type": "password", "description": "Clave usada para los embeddings y el modelo de chat."},
    {"key": "PDF_DIRECTORY", "label": "PDF_DIRECTORY", "type": "text", "description": "Ruta donde se leen los PDFs."},
    {"key": "MODEL_NAME", "label": "MODEL_NAME", "type": "text", "description": "Modelo de chat para las respuestas."},
    {"key": "TEMPERATURE", "label": "TEMPERATURE", "type": "number", "description": "Creatividad del modelo (0-1)."},
    {"key": "MAX_TOKENS", "label": "MAX_TOKENS", "type": "number", "description": "Límite de tokens en la respuesta."},
    {"key": "CHUNK_SIZE", "label": "CHUNK_SIZE", "type": "number", "description": "Tamaño de cada fragmento al indexar."},
    {"key": "CHUNK_OVERLAP", "label": "CHUNK_OVERLAP", "type": "number", "description": "Solape entre fragmentos."},
    {"key": "MAX_PAGES", "label": "MAX_PAGES", "type": "number", "description": "Páginas máximas a leer por PDF."},
    {"key": "RETRIEVER_K", "label": "RETRIEVER_K", "type": "number", "description": "Cantidad de fragmentos recuperados."},
    {"key": "SIMILARITY_THRESHOLD", "label": "SIMILARITY_THRESHOLD", "type": "number", "description": "Umbral de similitud para filtrar resultados."},
    {"key": "EMBEDDING_MODEL", "label": "EMBEDDING_MODEL", "type": "text", "description": "Modelo de embeddings."},
    {"key": "SYSTEM_PROMPT", "label": "SYSTEM_PROMPT", "type": "textarea", "description": "Mensaje de sistema para el asistente."},
    {"key": "PORT", "label": "PORT", "type": "number", "description": "Puerto del servidor Flask."},
    {"key": "WIDGET_API_KEY", "label": "WIDGET_API_KEY", "type": "text", "description": "API Key opcional para proteger el widget."}
]

# Global RAG Service
rag_service = None

def get_config():
    """Construye el diccionario de configuración desde las variables de entorno."""
    return {
        "BASE_DIR": BASE_DIR,
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        "PDF_DIRECTORY": os.getenv('PDF_DIRECTORY', './pdfs'),
        "MAX_PAGES": os.getenv('MAX_PAGES', 10),
        "CHUNK_SIZE": os.getenv('CHUNK_SIZE', 500),
        "CHUNK_OVERLAP": os.getenv('CHUNK_OVERLAP', 200),
        "MODEL_NAME": os.getenv('MODEL_NAME', 'gpt-4o-mini'),
        "TEMPERATURE": os.getenv('TEMPERATURE', 0.0),
        "MAX_TOKENS": os.getenv('MAX_TOKENS', 300),
        "SYSTEM_PROMPT": os.getenv('SYSTEM_PROMPT', "Eres un asistente útil."),
        "RETRIEVER_K": os.getenv('RETRIEVER_K', 3),
        "PORT": int(os.getenv('PORT', 9994)),
        "WIDGET_API_KEY": os.getenv('WIDGET_API_KEY', '')
    }

def read_env_values():
    """Devuelve un dict con los valores actuales del .env."""
    env_values = {}
    if not ENV_FILE.exists():
        return env_values

    with open(ENV_FILE, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_values[key.strip()] = value.strip()
    return env_values

def update_env_file(updates):
    """
    Actualiza el archivo .env preservando comentarios y orden.
    """
    existing_lines = []
    seen_keys = set()

    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            existing_lines = f.read().splitlines()

    new_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue

        key, _ = line.split("=", 1)
        key = key.strip()

        if key in updates:
            if key in seen_keys:
                continue
            new_lines.append(f"{key}={updates[key]}")
            seen_keys.add(key)
        else:
            if key in seen_keys:
                continue
            new_lines.append(line)
            seen_keys.add(key)

    for key, value in updates.items():
        if key not in seen_keys:
            new_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n")

    # Recargar variables y reiniciar servicio
    load_dotenv(ENV_FILE, override=True)
    initialize_service()

def initialize_service():
    """Inicializa o reinicia el servicio RAG."""
    global rag_service
    try:
        config = get_config()
        rag_service = RAGService(config)
        rag_service.initialize()
    except Exception as e:
        logger.critical(f"Failed to initialize RAG Service: {e}")

@app.before_request
def ensure_service_initialized():
    """Garantiza que el servicio esté listo."""
    if request.path.startswith("/api/env") or request.path.startswith("/env-admin"):
        return
    global rag_service
    if rag_service and rag_service.initialized:
        return
    initialize_service()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/env-admin')
def env_admin():
    current_env = read_env_values()
    filtered_env = {field["key"]: current_env.get(field["key"], "") for field in ENV_FIELDS}
    return render_template('env_admin.html', env_schema=ENV_FIELDS, env_data=filtered_env)

@app.route('/api/env', methods=['GET', 'POST'])
def env_api():
    if request.method == 'GET':
        current_env = read_env_values()
        filtered_env = {field["key"]: current_env.get(field["key"], "") for field in ENV_FIELDS}
        return jsonify({"env": filtered_env, "schema": ENV_FIELDS})

    payload = request.json or {}
    updates = payload.get("env")

    if not isinstance(updates, dict):
        return jsonify({"error": "Payload inválido"}), 400

    sanitized_updates = {str(k): "" if v is None else str(v) for k, v in updates.items()}

    try:
        update_env_file(sanitized_updates)
        return jsonify({"status": "ok", "saved": sanitized_updates})
    except Exception as e:
        logger.error(f"Error updating .env: {e}")
        return jsonify({"error": "No se pudo actualizar el .env"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='ok'), 200

@app.route('/chat', methods=['GET', 'POST', 'OPTIONS', 'HEAD'])
def chat():
    if request.method in ('OPTIONS', 'HEAD'):
        return ('', 204)
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'message': 'Use POST with JSON {prompt, history} to chat.'})

    data = request.json or {}
    prompt = data.get('prompt', '')
    history = data.get('history', [])

    config = get_config()
    widget_key = config.get('WIDGET_API_KEY')
    
    if widget_key:
        provided_key = request.headers.get('x-api-key')
        if provided_key != widget_key:
            return jsonify({'error': 'Unauthorized'}), 401

    if not prompt:
        return jsonify({'error': 'Missing required field: prompt'}), 400

    try:
        response_text = rag_service.answer_question(prompt, history)
        return jsonify({
            "choices": [{
                "message": {"content": response_text}
            }]
        })
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
    response.headers.add('Vary', 'Origin')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, x-api-key')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, HEAD')
    return response

if __name__ == '__main__':
    initialize_service()
    port = int(os.getenv('PORT', 9994))
    app.run(debug=True, host='0.0.0.0', port=port)
