import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict
from embeddings.embedder import STEMEmbedder
from data_processing.document_loader import DocumentLoader
from graph_db.neo4j_manager import Neo4jManager

load_dotenv()

class STEMRagSystem:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing in .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        self.embedder = STEMEmbedder()
        self.neo4j_manager = Neo4jManager()
        self.setup_complete = False

    def setup(self, force_rebuild=False):
        if not force_rebuild and self.embedder.load_embeddings():
            self.embedder.build_faiss_index()
            self.setup_complete = True
            print("RAG setup from stored embeddings")
            return
        loader = DocumentLoader()
        documents = loader.load_documents()
        if not documents:
            raise ValueError("No documents loaded")
        self.embedder.encode_documents(documents)
        self.embedder.build_faiss_index()
        self.embedder.save_embeddings()
        self.setup_complete = True
        print("RAG setup complete")

    def create_prompt(self, question: str, context_docs: List[Dict]) -> str:
        context_texts = "\n\n".join([f"Doc {i+1} ({doc['topic']}):\n{doc['text']}" for i, doc in enumerate(context_docs)])
        prompt = f"""
You are a STEM educational assistant. Use the following context to answer the question.
If not answerable, say so.

CONTEXT:
{context_texts}

QUESTION:
{question}

ANSWER:
"""
        return prompt.strip()

    def ask_question(self, question: str, top_k: int = 3) -> Dict:
        if not self.setup_complete:
            raise RuntimeError("Call setup() before asking questions.")

        vector_results = self.embedder.similarity_search(question, top_k=top_k)
        keywords = [kw for kw in ["Arduino", "Raspberry Pi", "LED", "Ohm's Law", "Sensor", "Actuator", "Microcontroller"] if kw.lower() in question.lower()]
        
        graph_results = []
        for keyword in keywords:
            graph_results.extend(self.neo4j_manager.find_chunks_for_concept(keyword))

        seen_texts = set()
        combined_context = []
        for doc in vector_results:
            if doc['text'] not in seen_texts:
                combined_context.append(doc)
                seen_texts.add(doc['text'])
        for doc in graph_results:
            if doc['text'] not in seen_texts:
                combined_context.append(doc)
                seen_texts.add(doc['text'])

        combined_context = combined_context[:top_k]

        prompt = self.create_prompt(question, combined_context)
        
        try:
            response = self.model.generate_content(prompt)
            return {
                "question": question,
                "answer": response.text,
                "context_docs": combined_context,
            }
        except Exception as e:
            return {
                "question": question,
                "answer": "Error generating response: " + str(e),
                "context_docs": combined_context,
            }

if __name__ == "__main__":
    rag = STEMRagSystem()
    rag.setup()
    q = "How do I program an LED to blink with Arduino?"
    res = rag.ask_question(q)
    print(res['answer'])
