# src/models/rag_system.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict
from src.embeddings.embedder import STEMEmbedder
from src.data_processing.document_loader import DocumentLoader

load_dotenv()

class STEMRagSystem:
    """RAG system for STEM learning assistance"""
    
    def __init__(self):
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize embedder
        self.embedder = STEMEmbedder()
        self.setup_complete = False
    
    def setup(self, force_rebuild: bool = False):
        """Setup the RAG system by loading or creating embeddings"""
        print("🚀 Setting up STEM RAG System...")
        
        # Try to load existing embeddings
        if not force_rebuild and self.embedder.load_embeddings():
            self.embedder.build_faiss_index()
            self.setup_complete = True
            print("✅ RAG system ready using existing embeddings!")
            return
        
        # Load and process documents
        print("📚 Loading documents...")
        loader = DocumentLoader()
        documents = loader.load_documents()
        
        if not documents:
            raise ValueError("No documents found! Please add .txt files to data/raw_documents/")
        
        # Generate embeddings
        self.embedder.encode_documents(documents)
        self.embedder.build_faiss_index()
        self.embedder.save_embeddings()
        
        self.setup_complete = True
        print("✅ RAG system setup complete!")
    
    def create_prompt(self, question: str, context_docs: List[Dict]) -> str:
        """Create prompt for Gemini with context and question"""
        context_text = "\n\n".join([
            f"Document {i+1} (Topic: {doc['topic']}):\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        prompt = f"""You are a helpful STEM learning assistant. Use the provided context to answer the student's question. 
If the context doesn't contain enough information, say so clearly.

CONTEXT:
{context_text}

STUDENT QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, educational answer suitable for students
- Use examples from the context when possible
- If mentioning specific components or concepts, explain them briefly
- If the question cannot be answered from the context, suggest related topics that might help

ANSWER:"""
        
        return prompt
    
    def ask_question(self, question: str, num_context_docs: int = 3) -> Dict:
        """Ask a question and get RAG-powered answer"""
        if not self.setup_complete:
            raise ValueError("RAG system not setup. Call setup() first.")
        
        print(f"❓ Question: {question}")
        
        # Retrieve relevant documents
        print("🔍 Finding relevant context...")
        relevant_docs = self.embedder.similarity_search(question, top_k=num_context_docs)
        
        if not relevant_docs:
            return {
                'question': question,
                'answer': 'No relevant information found in the knowledge base.',
                'context_docs': [],
                'error': 'No context found'
            }
        
        # Create prompt
        prompt = self.create_prompt(question, relevant_docs)
        
        try:
            # Generate answer using Gemini
            print("🤖 Generating answer...")
            response = self.model.generate_content(prompt)
            
            return {
                'question': question,
                'answer': response.text,
                'context_docs': relevant_docs,
                'num_context_docs': len(relevant_docs)
            }
            
        except Exception as e:
            return {
                'question': question,
                'answer': f'Error generating response: {str(e)}',
                'context_docs': relevant_docs,
                'error': str(e)
            }

# Test the RAG system
if __name__ == "__main__":
    # Create and setup RAG system
    rag = STEMRagSystem()
    rag.setup()
    
    # Test questions
    test_questions = [
        "How do I make an LED blink with Arduino?",
        "What is Ohm's Law and how do I use it?",
        "Explain how ultrasonic sensors work in robotics",
        "What's the difference between Arduino and Raspberry Pi?",
        "How do servo motors work?"
    ]
    
    print("\n" + "="*50)
    print("🎓 TESTING STEM LEARNING ASSISTANT")
    print("="*50)
    
    for question in test_questions:
        print(f"\n❓ {question}")
        print("-" * len(question))
        
        result = rag.ask_question(question)
        
        if 'error' not in result:
            print(f"🤖 Answer: {result['answer'][:200]}...")
            print(f"📚 Used {result['num_context_docs']} context documents")
        else:
            print(f"❌ Error: {result['error']}")
        
        print()
