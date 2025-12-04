
from PyPDF2 import PdfReader
import os

def test_extraction():
    pdf_path = "pdfs/GuiaParaEntenderAlGato.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            print(f"Number of pages: {len(reader.pages)}")
            
            # Print first 3 pages
            for i in range(min(3, len(reader.pages))):
                print(f"--- Page {i+1} ---")
                text = reader.pages[i].extract_text()
                print(f"Length: {len(text)}")
                print(text[:500])
                print("----------------")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_extraction()
