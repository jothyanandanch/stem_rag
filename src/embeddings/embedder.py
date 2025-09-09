
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import pickle
import os

class STEMEmbedder:    
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        print(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        
        os.makedirs("data/embeddings", exist_ok=True)
    
    def encode_documents(self, documents):
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32
            )
        
        self.embeddings = embeddings
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self):
        
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run encode_documents first.")
        
        print("Building FAISS index for fast similarity search...")
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(self.embeddings)
        
        self.index.add(self.embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def similarity_search(self, query, top_k= 5):
        if self.index is None:
            raise ValueError("No index found. Run build_faiss_index first.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save_embeddings(self, filepath: str = "data/embeddings/stem_embeddings.pkl"):
        data = {
            'embeddings': self.embeddings,
            'documents': self.documents,
            'model_name': self.model.get_sentence_embedding_dimension()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f" Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str = "data/embeddings/stem_embeddings.pkl"):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data['embeddings']
            self.documents = data['documents']
            
            print(f"📁 Loaded embeddings: {self.embeddings.shape}")
            return True
            
        except FileNotFoundError:
            print(f"❌ No saved embeddings found at {filepath}")
            return False


if __name__ == "__main__":
    from src.data_processing.document_loader import DocumentLoader
    
    # Load documents
    loader = DocumentLoader()
    docs = loader.load_documents()
    
    if docs:
        # Create embeddings
        embedder = STEMEmbedder()
        embedder.encode_documents(docs)
        embedder.build_faiss_index()
        
        # Test similarity search
        test_query = "How do I control a servo motor with Arduino?"
        results = embedder.similarity_search(test_query, top_k=3)
        
        print(f"\n🔍 Query: '{test_query}'")
        print("📋 Top Results:")
        for result in results:
            print(f"  {result['rank']}. [{result['topic']}] Score: {result['similarity_score']:.3f}")
            print(f"     {result['text'][:100]}...")
            print()
        
        # Save embeddings
        embedder.save_embeddings()
