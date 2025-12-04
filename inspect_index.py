
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

def inspect_index():
    if not os.path.exists("faiss_index"):
        print("No FAISS index found.")
        return

    try:
        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        print(f"Index has {docsearch.index.ntotal} vectors.")
        
        # Print first 5 documents from the docstore
        print("\n--- First 5 Chunks in Docstore ---")
        for i, (id, doc) in enumerate(docsearch.docstore._dict.items()):
            if i >= 5: break
            print(f"Chunk {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print("----------------")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_index()
