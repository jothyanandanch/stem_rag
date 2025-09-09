# Test all imports work with your versions
def test_imports():
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import sentence_transformers
        print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
        
        import faiss
        print(f"✅ FAISS: Available")
        
        import neo4j
        print(f"✅ Neo4j: {neo4j.__version__}")
        
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
        
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
        
        import google.generativeai as genai
        print(f"✅ Google Generative AI: Available")
        
        print("\n🎉 All imports successful! Ready for Phase 2.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
