# 🧠 PDF Chat API with LangChain, OpenAI & FAISS

This project is a Flask-based API to query local PDF documents with natural language. It uses OpenAI LLMs, LangChain, and FAISS for semantic search over your document chunks.

## 🚀 Features

- 📄 Reads and processes local PDF documents
- 🤖 Embeds content using OpenAI embeddings
- 📚 Builds a FAISS vector index for fast semantic search
- 🧠 Answers user questions based on those PDFs
- 🛡️ Includes a health check endpoint
- 🔐 Configurable via `.env` file (API key, chunking, model, widget API key, etc.)

---

## 📦 Requirements

- Python 3.9+ (tested with 3.9)
- OpenAI account (API key)
- Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_key
PDF_DIRECTORY=./pdfs
MAX_PAGES=50
CHUNK_SIZE=500
CHUNK_OVERLAP=0
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.5
MAX_TOKENS=300
PORT=9994
RETRIEVER_K=3
WIDGET_API_KEY=optional_shared_secret
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place your PDF files inside the directory specified in `PDF_DIRECTORY`.

### Adding New Documents

1. Copy your new PDF into the `pdfs/` folder.
2. Restart the server (`python app.py`).
3. The server detects new files, embeds them, and updates the FAISS index.  
   _Note_: Do not edit `processed_files.json`; it tracks processed PDFs.

---

## 🧠 How It Works

- Loads PDFs from the configured folder
- Loads PDFs from the configured folder
- Splits text with `RecursiveCharacterTextSplitter`
- Embeds content via `OpenAIEmbeddings`
- Stores/loads a FAISS index
- Uses `ConversationalRetrievalChain` to answer prompts at `/chat`

---

## 🚀 Run the Server

```bash
python app.py
```

The app runs on `http://localhost:9994`.

---

## 🛠️ API Endpoints

### `GET /health`
Quick health check:
```bash
curl http://localhost:9994/health
```

### `POST /chat`
Send a prompt and history:

```json
{
  "prompt": "What does the document say about data privacy?",
  "history": []  // optional, conversational pairs
}
```

Returns:

```json
{
  "choices": [
    {
      "message": {
        "content": "The document states that data privacy is ensured by..."
      }
    }
  ]
}
```

---

## 📁 FAISS Index

Once built, the FAISS index is saved locally in `faiss_index/` for future reuse.

---

## 🔒 Widget API key (optional)

If you expose the `/chat` endpoint publicly, set `WIDGET_API_KEY` in `.env`. The widget will send it as `x-api-key` when you set the same value in `window.RAG_WIDGET_CONFIG.apiKey` inside `templates/index.html`.

---

## 📝 License

MIT License — do whatever you want, just don’t forget the zorro 🦊

---

## ✨ Author

Oscar Aguilera  
GitHub: [@agui1era](https://github.com/agui1era)
