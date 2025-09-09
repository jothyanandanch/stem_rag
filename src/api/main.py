from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import sys
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag_system import STEMRagSystem

app = FastAPI(title="STEM Learning Assistant", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    return index_path.read_text()

rag_system = STEMRagSystem()

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class AnswerResponse(BaseModel):
    question: str
    answer: str
    context_docs: List[str]

@app.on_event("startup")
async def startup():
    rag_system.setup()

@app.get("/")
async def home():
    return {"status": "ready"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    try:
        result = rag_system.ask_question(req.question, req.top_k)
        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            context_docs=[d["text"][:200] + "..." for d in result["context_docs"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
