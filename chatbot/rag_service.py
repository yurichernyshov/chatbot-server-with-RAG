import logging
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from config import Config


logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.config = Config()
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        """Инициализация всех компонентов"""
        logger.info("Инициализация RAG сервиса...")
        
        # Инициализация эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Эмбеддинги загружены: {self.config.EMBEDDING_MODEL}")
        
        # Инициализация векторной базы
        from chromadb.config import Settings

        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory="/app/chroma_db",
            client_settings=Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=self.config.CHROMA_HOST,
                chroma_server_http_port=self.config.CHROMA_PORT,
                persist_directory="./app/chroma_db")
        )
        logger.info("Векторная база данных подключена")
        
        # Инициализация LLM
        self.llm = Ollama(
            model=self.config.LLM_MODEL,
            base_url=self.config.OLLAMA_HOST
        )
        logger.info(f"LLM подключен: {self.config.LLM_MODEL}")
        
        # Инициализация QA цепи
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        logger.info("RAG сервис готов к работе")

    def load_documents(self, documents_path: Optional[str] = None) -> int:
        """Загрузка документов в векторную базу"""
        path = documents_path or self.config.DOCUMENTS_PATH
        logger.info(f"Загрузка документов из: {path}")
        
        try:
            loader = DirectoryLoader(
                path=path,
                glob="**/*.txt",
                show_errors=True
            )
            documents = loader.load()
            
            if not documents:
                logger.warning("Документы не найдены")
                return 0
            
            # Разделение документов
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Добавление в векторную базу
            self.vectorstore.add_documents(chunks)
            
            logger.info(f"Загружено {len(chunks)} чанков")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")
            raise

    def query(self, question: str) -> dict:
        """Поиск ответа на вопрос"""
        logger.info(f"Запрос: {question}")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            response = {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "status": "success"
            }
            
            logger.info("Запрос успешно обработан")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": "Произошла ошибка при обработке запроса",
                "sources": [],
                "status": "error",
                "error": str(e)
            }

    def get_stats(self) -> dict:
        """Получение статистики"""
        try:
            count = self.vectorstore._collection.count()
            return {
                "documents_count": count,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "llm_model": self.config.LLM_MODEL,
                "status": "healthy"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

