import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
from config import Config
from rag_service import RAGService

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="API для чатбота с RAG и векторной базой данных",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация сервиса
rag_service = RAGService()

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    status: str

class StatsResponse(BaseModel):
    documents_count: int
    embedding_model: str
    llm_model: str
    status: str

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    stats = rag_service.get_stats()
    return stats

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Получение статистики"""
    return rag_service.get_stats()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Обработка запроса к чатботу"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    result = rag_service.query(request.question)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Загрузка документов для индексации"""
    try:
        uploaded_count = 0
        
        for file in files:
            file_path = os.path.join(Config.DOCUMENTS_PATH, file.filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_count += 1
            logger.info(f"Файл загружен: {file.filename}")
        
        # Переиндексация документов
        chunks_count = rag_service.load_documents()
        
        return {
            "status": "success",
            "files_uploaded": uploaded_count,
            "chunks_indexed": chunks_count
        }
        
    except Exception as e:
        logger.error(f"Ошибка загрузки документов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
async def reindex():
    """Переиндексация всех документов"""
    try:
        chunks_count = rag_service.load_documents()
        return {
            "status": "success",
            "chunks_indexed": chunks_count
        }
    except Exception as e:
        logger.error(f"Ошибка переиндексации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.CHATBOT_HOST,
        port=Config.CHATBOT_PORT
    )
