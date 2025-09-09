import os
import glob
from typing import List, Dict
from .document_processor import DocumentProcessor

CONCEPT_KEYWORDS = [
    "Arduino", "Raspberry Pi", "LED", "Ohm's Law",
    "Sensor", "Actuator", "Microcontroller"
]

def extract_concepts_from_text(text: str, concept_keywords=CONCEPT_KEYWORDS) -> List[str]:
    found = []
    text_lower = text.lower()
    for kw in concept_keywords:
        if kw.lower() in text_lower and kw not in found:
            found.append(kw)
    return found

class DocumentLoader:
    def __init__(self, documents_path="data/raw_documents"):
        self.documents_path = documents_path
        self.processor = DocumentProcessor()

    def load_documents(self) -> List[Dict]:
        documents = []
        txt_files = glob.glob(os.path.join(self.documents_path, "*.txt"))

        for file_path in txt_files:
            filename = os.path.basename(file_path)
            topic = filename.replace('.txt', '').replace('_', ' ').title()

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            chunks = self.processor.smart_chunk_text(content)
            for chunk in chunks:
                chunk['source_file'] = filename
                chunk['topic'] = topic
                chunk['file_path'] = file_path
                # Extract concepts mentioned
                chunk['concepts'] = extract_concepts_from_text(chunk['text'])
            documents.extend(chunks)

        return documents

if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_documents()
    print(f"Loaded {len(docs)} chunks from documents.")

    # Show first chunk sample
    print(docs[0])
