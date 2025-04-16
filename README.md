
# 🧠 PDF Chat API with LangChain, OpenAI & FAISS

This project is a Flask-based API that allows users to query local PDF documents using natural language.  
It uses OpenAI's LLMs, LangChain for chaining, and FAISS for semantic vector search over the document content.

## 🚀 Features

- 📄 Reads and processes local PDF documents
- 🤖 Embeds content using OpenAI embeddings
- 📚 Creates a FAISS vector index for fast semantic search
- 🧠 Answers user questions based on the content of those PDFs
- 🛡️ Includes a health check endpoint
- 🔐 Configurable via `.env` file

---

## 📦 Requirements

- Python 3.10+
- OpenAI account (API key)
- Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_key
PDF_DIRECTORY=./pdfs
MAX_PAGES=10
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

---

## 🧠 How It Works

- Loads PDFs from the configured folder
- Extracts and splits text using LangChain’s `CharacterTextSplitter`
- Embeds the content using `OpenAIEmbeddings`
- Stores and/or loads a FAISS index
- Initializes a RetrievalQA system
- Responds to user prompts at `/chat` endpoint

---

## 🚀 Run the Server

```bash
python app.py
# or if the file has a different name, e.g.:
python main.py
```

The app runs on: `http://localhost:8080`

---

## 🛠️ API Endpoints

### `GET /health`

Check if the API is alive:
```bash
curl http://localhost:8080/health
```

### `POST /chat`

Send a prompt and get an answer based on the PDFs:
```json
{
  "prompt": "What does the document say about data privacy?"
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

## 📝 License

MIT License — do whatever you want, just don’t forget the zorro 🦊

---

## ✨ Author

Oscar Aguilera  
GitHub: [@agui1era](https://github.com/agui1era)
