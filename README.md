# chatbot-server-with-RAG
Example of chat bot server with RAG for experiments

В проекте создается микросервисная архитектура для чатбота на базе LLM, который использует RAG и векторную базу данных. 
Отдельные docker контейнеры: 
- ollama,  
- чат-бота на python и langchain с простым графическим интерфейсом,
- векторной базы данных chromadb.

# Создание директорий
mkdir -p data/documents

# Загрузка модели в Ollama (после запуска)
docker exec -it ollama-service ollama pull qwen3:4b

# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down

# Остановка с удалением томов
docker-compose down -v

