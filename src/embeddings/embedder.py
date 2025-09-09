import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict
import pickle
import os

class STEMEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        os.makedirs("data/embeddings", exist_ok=True)

    def encode_documents(self, documents: List[Dict]) -> np.ndarray:
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
        self.embeddings = embeddings
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def build_faiss_index(self):
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run encode_documents first.")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            raise ValueError("No index found. Run build_faiss_index first.")
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                results.append(doc)
        return results

    def save_embeddings(self, filepath="data/embeddings/stem_embeddings.pkl"):
        data = {'embeddings': self.embeddings, 'documents': self.documents}
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved embeddings to {filepath}")

    def load_embeddings(self, filepath="data/embeddings/stem_embeddings.pkl") -> bool:
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.documents = data['documents']
            print(f"Loaded embeddings: {self.embeddings.shape}")
            return True
        except FileNotFoundError:
            print(f"No embeddings found at {filepath}")
            return False

if __name__ == "__main__":
    from src.data_processing.document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_documents()
    embedder = STEMEmbedder()
    embedder.encode_documents(docs)
    embedder.build_faiss_index()
    results = embedder.similarity_search("How do I make an LED blink?")
    for r in results:
        print(r['text'][:200])
