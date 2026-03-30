# Связка Guard (Next.js) и Python-анализатора

## Роли компонентов

| Компонент | Назначение |
|-----------|------------|
| **Guard** (`guard-main/`) | Веб: загрузка документов, MinHash по локальной SQLite-базе, PDF, админка. |
| **Python API** (`api_server.py` + `worker.py`) | Векторный плагиат (Qdrant + rubert-tiny2) и эвристика «AI-%» (ruGPT3 small). |
| **Qdrant** | Хранение эмбеддингов чанков; при анализе ищутся близкие векторы, затем новые чанки добавляются в коллекцию. |

MinHash в Guard и ML-анализ в Python **дополняют** друг друга: в API проверки отдаются оба набора метрик (если задан `ANALYSIS_SERVICE_URL`).

---

## API-контракт (Python → потребитель Guard)

Base URL задаётся в Guard как `ANALYSIS_SERVICE_URL` (без завершающего `/`).

### `GET /health`

**Ответ 200**

```json
{ "status": "ok", "worker_loaded": true }
```

### `POST /v1/analyze`

**Заголовки**

| Заголовок | Обязателен | Описание |
|-----------|------------|----------|
| `Content-Type: application/json` | да | |
| `X-API-Key` | нет | Если на сервисе задан `ANALYSIS_API_KEY`, без совпадения вернётся `401`. |

**Тело запроса**

```json
{
  "content": "полный текст документа после нормализации на стороне Guard",
  "filename": "имя_файла.docx",
  "document_id": 123
}
```

- `document_id` опционален, для логов; не обязан совпадать с реальным ID в SQLite Guard.
- Текст **должен** совпадать с тем, что Guard использует для MinHash: тот же пайплайн, что `normalizeContentForCheck` в `/api/check` и `/api/upload`.

**Ответ 200**

```json
{
  "plagiarism_percent": 12.5,
  "ai_percent": 34.2
}
```

| Поле | Тип | Смысл |
|------|-----|--------|
| `plagiarism_percent` | number | Доля чанков с hit в Qdrant (порог в воркере), 0–100. |
| `ai_percent` | number | Усреднённая эвристика «похожести на AI» по чанкам, 0–100. |

**Ошибки**

- `401` — неверный/отсутствующий ключ при включённом `ANALYSIS_API_KEY`.
- `503` — воркер не инициализирован.
- `422` — невалидное тело (Pydantic).

---

## Переменные окружения

### Python (`api_server.py`)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `QDRANT_HOST` | `localhost` | |
| `QDRANT_PORT` | `6333` | HTTP-порт Qdrant. |
| `QDRANT_COLLECTION` | `university_docs` | Имя коллекции (должна существовать или быть создана скриптом). |
| `ANALYSIS_API_KEY` | _(пусто)_ | Если задан, Guard должен слать тот же ключ в `X-API-Key`. |
| `ANALYSIS_SQLITE_PATH` | `showcase.db` | Путь для поля в `AntiPlagiarismWorker` (очередь воркера не используется API; значение почти не влияет на HTTP API). |

### Guard (Next.js)

| Переменная | Описание |
|------------|----------|
| `ANALYSIS_SERVICE_URL` | URL Python API, например `http://127.0.0.1:8765`. Если не задан — ML-метки не запрашиваются, приложение работает только на MinHash. |
| `ANALYSIS_SERVICE_API_KEY` | Должен совпадать с `ANALYSIS_API_KEY` на Python, если тот включён. |
| `SQLITE_PATH` | _(опционально)_ Общий путь к `app.db` при вынесенном диске. |

---

## Поток данных в Guard после доработки

1. **`POST /api/check`**  
   - MinHash по SQLite, как раньше.  
   - Параллельно (после нормализации текста) вызывается `POST /v1/analyze`.  
   - В JSON добавляются `mlPlagiarismPercent` и `mlAiPercent`, если сервис ответил.

2. **`POST /api/upload`**  
   - В форму можно передать `plagiarism_percent_ml` и `ai_percent_ml` (строки с числами), чтобы не дублировать вызов после проверки.  
   - Если хотя бы одно из полей отсутствует, а `ANALYSIS_SERVICE_URL` задан и текст ≥ 50 символов — сервер сам вызывает `/v1/analyze` и пишет результат в SQLite.

3. **SQLite**  
   В таблице `documents` добавлены столбцы:  
   - `plagiarism_percent_ml`  
   - `ai_percent_ml`  

   Миграция выполняется при старте приложения в `lib/sqlite.ts` (`migrate`).

---

## Развёртывание (кратко)

1. Поднять Qdrant (например Docker на `6333`).  
2. Создать коллекцию (один раз):

   ```bash
   cd antiplagiarism
   pip install -r requirements.txt
   export QDRANT_URL=http://localhost:6333
   export QDRANT_COLLECTION=university_docs
   python init_qdrant_collection.py
   ```

3. Запустить Python API (модели подтянутся при старте, нужен интернет или кеш HuggingFace):

   ```bash
   export QDRANT_COLLECTION=university_docs
   # опционально: export ANALYSIS_API_KEY=длинная_случайная_строка
   uvicorn api_server:app --host 0.0.0.0 --port 8765
   ```

4. Запустить Guard с переменными:

   ```bash
   export ANALYSIS_SERVICE_URL=http://127.0.0.1:8765
   # export ANALYSIS_SERVICE_API_KEY=... если включили ANALYSIS_API_KEY
   cd guard-main && npm run dev
   ```

На прод-сервере удобно вынести Python и Qdrant в `docker-compose` и прописать `ANALYSIS_SERVICE_URL` на внутренний hostname сервиса (например `http://analysis:8765`).

---

## Два проекта на сервере: Docker (типовой сценарий)

### 1) Как запустить «второй проект» (ML + Qdrant)

В каталоге `antiplagiarism` (где лежат `worker.py`, `docker-compose.ml.yml`):

```bash
docker compose -f docker-compose.ml.yml up -d --build
```

Поднимутся:

- **Qdrant** — порт **6333** на хосте (данные в volume `qdrant_storage`).
- **analysis** — HTTP API на порту **8765** (`/v1/analyze`, `/health`).  
  При старте ждёт Qdrant, создаёт коллекцию при необходимости, затем запускает uvicorn.  
  Первый запрос может быть долгим: скачивание моделей с Hugging Face (нужен исходящий интернет или заранее смонтированный кеш).

Сборка образа тяжёлая (PyTorch и зависимости). На слабом сервере заложите место на диск и RAM.

### 2) Как связать со Guard

Связь уже реализована в коде Guard через переменные окружения (см. выше). Достаточно задать **`ANALYSIS_SERVICE_URL`** так, чтобы контейнер `app` Guard **достучался** до сервиса `analysis`.

**Вариант A — общая Docker-сеть (удобно на Linux)**

Один раз создайте сеть (имя должно совпадать с `docker-compose.ml.yml`, сеть `antiplag`):

```bash
docker network create antiplag
```

Файл `docker-compose.ml.yml` уже подключает к ней `qdrant` и `analysis`.  
В `guard-main/docker-compose.yml` у сервиса **`app`** добавьте сеть `antiplag` и переменную:

```yaml
  app:
    environment:
      ANALYSIS_SERVICE_URL: "http://analysis:8765"
      # при ANALYSIS_API_KEY в ML-стеке:
      # ANALYSIS_SERVICE_API_KEY: "тот_же_секрет"
    networks:
      - plagiarismguard-network
      - antiplag

networks:
  plagiarismguard-network:
    driver: bridge
  antiplag:
    external: true
```

Тогда Guard обращается к контейнеру `antiplag-analysis` по DNS-имени **`analysis`**.

**Вариант B — без общей сети (проще мысленно, хуже для изоляции)**

Оставьте ML-compose как есть (порт **8765** проброшен на хост). В Guard укажите URL до хоста, например:

- `ANALYSIS_SERVICE_URL=http://172.17.0.1:8765` (часто шлюз Docker на Linux),
- или реальный IP сервера / `host.docker.internal` (Docker Desktop на Mac/Windows).

Точное значение зависит от ОС и сетевых политик; проверьте с хоста: `curl http://127.0.0.1:8765/health`.

### Порядок запуска

1. Поднять ML-стек (`docker-compose.ml.yml`), дождаться готовности (`curl .../health`).
2. Поднять Guard с **`ANALYSIS_SERVICE_URL`**.  
   Без этой переменной Guard работает **только** по MinHash, Python не вызывается.

Полный API-контракт и поля env — в разделах выше.

---

## Версионирование

Текущая версия HTTP API: **`1.0`** (префикс `/v1/analyze`). Несовместимые изменения — новый путь (`/v2/...`) или поле `api_version` в ответе (при необходимости расширить контракт позже).
