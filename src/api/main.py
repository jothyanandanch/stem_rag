# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag_system import STEMRagSystem

app = FastAPI(title="STEM Learning Assistant", version="1.0.0")

# Initialize RAG system
rag_system = STEMRagSystem()

class QuestionRequest(BaseModel):
    question: str
    num_context_docs: int = 3

class AnswerResponse(BaseModel):
    question: str
    answer: str
    num_context_docs: int
    context_used: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    print("🚀 Starting STEM Learning Assistant API...")
    rag_system.setup()
    print("✅ API is ready!")

@app.get("/")
async def root():
    return {"message": "STEM Learning Assistant API", "status": "ready"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an AI-powered answer"""
    try:
        result = rag_system.ask_question(
            question=request.question,
            num_context_docs=request.num_context_docs
        )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Extract context information
        context_used = [
            f"{doc['topic']} (Score: {doc['similarity_score']:.3f})"
            for doc in result['context_docs']
        ]
        
        return AnswerResponse(
            question=result['question'],
            answer=result['answer'],
            num_context_docs=result['num_context_docs'],
            context_used=context_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "system": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
