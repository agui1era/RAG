import os
import json
import logging
from pathlib import Path
import tiktoken
from PyPDF2 import PdfReader

# Updated imports to silence deprecation warnings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, config):
        self.config = config
        self.api_key = config.get('OPENAI_API_KEY')
        self.pdf_directory = Path(config.get('PDF_DIRECTORY', './pdfs'))
        self.base_dir = config.get('BASE_DIR')
        self.faiss_index_dir = self.base_dir / "faiss_index"
        self.processed_files_log = self.base_dir / "processed_files.json"
        
        # Resolve PDF directory if relative
        if not self.pdf_directory.is_absolute():
            self.pdf_directory = self.base_dir / self.pdf_directory

        self.max_pages = int(config.get('MAX_PAGES', 10))
        self.chunk_size = int(config.get('CHUNK_SIZE', 500))
        self.chunk_overlap = int(config.get('CHUNK_OVERLAP', 200))
        self.model_name = config.get('MODEL_NAME', 'gpt-4o-mini')
        self.temperature = float(config.get('TEMPERATURE', 0.0))
        self.max_tokens = int(config.get('MAX_TOKENS', 300))
        self.system_prompt = config.get('SYSTEM_PROMPT', "Eres un asistente útil.")
        self.retriever_k = int(config.get('RETRIEVER_K', 3))

        self.docsearch = None
        self.qa = None
        self.initialized = False

    def load_processed_files(self):
        """Carga la lista de archivos ya procesados."""
        if self.processed_files_log.exists():
            try:
                with open(self.processed_files_log, 'r') as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                logger.warning(f"{self.processed_files_log} is empty or corrupted. Starting with empty set.")
                return set()
        return set()

    def save_processed_files(self, processed_files):
        """Guarda la lista de archivos procesados."""
        with open(self.processed_files_log, 'w') as f:
            json.dump(list(processed_files), f)

    def load_pdfs_from_local(self, processed_files):
        """Carga y procesa solo los PDFs nuevos desde un directorio local."""
        new_text = ""
        new_files = []
        
        if not self.pdf_directory.exists():
            self.pdf_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.pdf_directory.resolve()}")
            return "", []

        for filepath in self.pdf_directory.glob("*.pdf"):
            filename = filepath.name
            if filename not in processed_files:
                try:
                    with open(filepath, 'rb') as f:
                        pdf_reader = PdfReader(f)
                        pdf_text = ""
                        for i, page in enumerate(pdf_reader.pages[:self.max_pages]):
                            pdf_text += page.extract_text()
                        new_text += pdf_text
                        new_text += f"\n\nEnd of document: {filename}\n\n"
                        logger.info(f"Loaded new PDF: {filename}, Pages: {min(self.max_pages, len(pdf_reader.pages))}")
                        new_files.append(filename)
                except Exception as e:
                    logger.error(f"Failed to read {filename}: {e}")
                    
        return new_text, new_files

    def count_tokens(self, text):
        """Cuenta los tokens en un texto usando tiktoken."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def embed_documents(self, docs, existing_index=None):
        """Crea embeddings de los documentos y los guarda/actualiza en FAISS."""
        try:
            embeddings = OpenAIEmbeddings(api_key=self.api_key)
            doc_objs = [Document(page_content=doc) for doc in docs]
            
            if existing_index:
                existing_index.add_documents(doc_objs)
                faiss_index = existing_index
                logger.info("Added new documents to existing FAISS index")
            else:
                faiss_index = FAISS.from_documents(doc_objs, embeddings)
            logger.info("Created new FAISS index")
                
            faiss_index.save_local(str(self.faiss_index_dir))
            logger.info(f"FAISS index saved locally at '{self.faiss_index_dir}'")
            return faiss_index
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def initialize(self):
        """Inicializa documentos desde FAISS y agrega nuevos si existen."""
        processed_files = self.load_processed_files()
        
        # 1. Intentar cargar el índice existente
        if processed_files:
            try:
                self.docsearch = FAISS.load_local(
                    str(self.faiss_index_dir), 
                    OpenAIEmbeddings(api_key=self.api_key), 
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS index loaded from local storage at {self.faiss_index_dir}")
            except Exception:
                self.docsearch = None
                logger.info("No existing FAISS index found or error loading it.")
        else:
            self.docsearch = None
            logger.info("Processed files list is empty. Starting with a fresh index.")

        # 2. Buscar archivos nuevos
        logger.info(f"Checking for new files in {self.pdf_directory.resolve()}...")
        new_pdf_text, new_files = self.load_pdfs_from_local(processed_files)
        
        if new_files:
            logger.info(f"Found {len(new_files)} new files to process.")
            total_tokens = self.count_tokens(new_pdf_text)
            logger.info(f"Total tokens in new files: {total_tokens}")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            docs = text_splitter.split_text(new_pdf_text)
            
            # 3. Actualizar o crear índice
            self.docsearch = self.embed_documents(docs, existing_index=self.docsearch)
            
            # 4. Actualizar registro de archivos procesados
            processed_files.update(new_files)
            self.save_processed_files(processed_files)
        else:
            logger.info("No new files found to process.")
            
        if self.docsearch is None:
            logger.warning("Index is empty. Creating placeholder.")
            embeddings = OpenAIEmbeddings(api_key=self.api_key)
            self.docsearch = FAISS.from_documents([Document(page_content="Placeholder")], embeddings)

        # Inicializar el modelo de lenguaje y el sistema de preguntas y respuestas
        llm = ChatOpenAI(
            api_key=self.api_key, 
            model_name=self.model_name, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens
        )
        
        template = f"""{{system_prompt}}

Context:
{{context}}

Question: {{question}}
Helpful Answer:"""
        
        final_template = template.format(system_prompt=self.system_prompt, context="{context}", question="{question}")
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=final_template)

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=self.docsearch.as_retriever(search_kwargs={"k": self.retriever_k}),
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        self.initialized = True
        logger.info("RAG Service initialization complete")

    def answer_question(self, prompt, history):
        if not self.qa:
            raise Exception("RAG Service not initialized")
            
        # Convertir historial al formato esperado por LangChain [(q, a), (q, a)]
        chat_history = []
        temp_q = None
        for msg in history:
            if msg.get('role') == 'user':
                temp_q = msg.get('content')
            elif msg.get('role') == 'assistant' and temp_q:
                chat_history.append((temp_q, msg.get('content')))
                temp_q = None
                
        result = self.qa({"question": prompt, "chat_history": chat_history})
        return result['answer']
