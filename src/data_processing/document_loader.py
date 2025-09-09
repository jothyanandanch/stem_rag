import os 
from typing import List, Dict
from document_processor import DocumentProcessor

class DocumentLoader:
    def __init__(self, directory="D:\\Educational\\Projects\\stem_learning_assistant\\data\\raw_documents\\"):
        self.directory = directory
        self.processor = DocumentProcessor()
        self.documents = []

    def load_documents(self) -> List[Dict]:
        documents = []

        txtfiles = [
            f for f in os.listdir(self.directory)
            if os.path.isfile(os.path.join(self.directory, f)) and f.endswith(".txt")
        ]

        print(f"Found {len(txtfiles)} documents")

        for filename in txtfiles:
            full_path = os.path.join(self.directory, filename)
            topic = filename.replace('.txt', '').replace('_', ' ')
            try:
                with open(full_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                CONCEPT_KEYWORDS = ["Arduino", "Raspberry Pi", "LED", "Ohm's Law", "Sensor", "Actuator", "Microcontroller"]
                chunks = self.processor.chunk_text(content)
                for chunk in chunks:
                    chunk['source_file'] = filename
                    chunk['topic'] = topic
                    chunk['filepath'] = full_path
                    chunk['concepts'] = extract_concepts_from_text(chunk['text'], CONCEPT_KEYWORDS)
                documents.extend(chunks)

                if chunks:
                    print(f"✅ First chunk from {filename}: {documents[0]}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

        print(f"\n🚀 Total chunks created: {len(documents)}")
        self.documents = documents
        return documents

    def get_document_stats(self, documents: List[Dict]) -> Dict:
        stats = {
            'total_chunks': len(documents),
            'topics': set(),
            'total_words': 0,
            'avg_chunk_size': 0
        }

        for doc in documents:
            stats['topics'].add(doc['topic'])
            stats['total_words'] += doc.get('word_count', 0)

        if documents:
            stats['avg_chunk_size'] = stats['total_words'] / len(documents)
            stats['topics'] = list(stats['topics'])

        return stats
    

    

def extract_concepts_from_text(text: str, concept_keywords: List[str]) -> List[str]:
    found = []
    text_lower = text.lower()
    for kw in concept_keywords:
        if kw.lower() in text_lower:
            found.append(kw)
    return found


if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_documents()
    
    stats = loader.get_document_stats(docs)
    print(f"\n📊 Document Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Topics covered: {', '.join(stats['topics'])}")
    print(f"Average chunk size: {stats['avg_chunk_size']:.1f} words")
