# Развёртывание на сервере: Guard + ML (Qdrant + Python API)

Краткий чеклист задач и команд. Детали контракта API и переменных — в [INTEGRATION.md](./INTEGRATION.md).

---

## Задачи перед запуском (чеклист)

- [ ] На сервере установлены **Docker** и **Docker Compose** (плагин `docker compose`).
- [ ] Есть **исходники** обоих проектов в одном родителе, например:
  - `.../antiplagiarism/` — Python, `docker-compose.ml.yml`, `worker.py`, `api_server.py`
  - `.../antiplagiarism/guard-main/` — Next.js, Dockerfile, `docker-compose.yml`
- [ ] Открыты/проксированы нужные **порты** (или только внутренняя сеть):
  - веб: как настроите в nginx (часто **3000** и/или **443**);
  - опционально снаружи: **8765** (ML API), **6333** (Qdrant) — на проде лучше не публиковать 6333 наружу.
- [ ] Есть **исходящий интернет** с сервера при первом запуске ML-контейнера (скачивание моделей Hugging Face), либо заранее подготовленный кеш моделей.
- [ ] Запас по **RAM/диску** под PyTorch и Qdrant (ориентир: несколько ГБ RAM, де десятки ГБ диска).

---

## 1. Скопировать проект на сервер

Пример (замените URL и каталог):

```bash
ssh user@YOUR_SERVER
sudo mkdir -p /opt/antiplag && sudo chown "$USER:$USER" /opt/antiplag
cd /opt/antiplag
git clone <URL_репозитория> repo
cd repo/antiplagiarism
```

Далее команды считаем, что текущий каталог — **`/opt/antiplag/repo/antiplagiarism`** (родитель `guard-main`).

---

## 2. Поднять ML-стек (Qdrant + анализ)

```bash
cd /opt/antiplag/repo/antiplagiarism

docker compose -f docker-compose.ml.yml build
docker compose -f docker-compose.ml.yml up -d
```

Проверка:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:6333/readyz
```

Ожидание: JSON с `"status":"ok"` на `/health` и ответ от Qdrant на `/readyz`. Первый запуск анализа может занять много времени из‑за моделей.

Логи при проблемах:

```bash
docker compose -f docker-compose.ml.yml logs -f analysis
docker compose -f docker-compose.ml.yml logs -f qdrant
```

Остановка / перезапуск:

```bash
docker compose -f docker-compose.ml.yml down
docker compose -f docker-compose.ml.yml up -d
```

---

## 3. Связать Guard с ML через Docker-сеть

Сеть **`antiplag`** создаётся при первом `up` файла `docker-compose.ml.yml` (имя задано в compose).

Отредактируйте **`guard-main/docker-compose.yml`**:

1. У сервиса **`app`** в **`networks`** добавьте **`antiplag`** (и оставьте **`plagiarismguard-network`**).
2. В секции **`networks`** внизу файла раскомментируйте **`antiplag`** с **`external: true`**.
3. У сервиса **`app`** в **`environment`** задайте:

```yaml
ANALYSIS_SERVICE_URL: "http://analysis:8765"
```

Опционально (если в `docker-compose.ml.yml` у сервиса `analysis` задан `ANALYSIS_API_KEY`):

```yaml
ANALYSIS_SERVICE_API_KEY: "<тот же секрет>"
```

Сохраните файл.

---

## 4. Поднять Guard (Next.js + nginx)

```bash
cd /opt/antiplag/repo/antiplagiarism/guard-main

# при необходимости создайте .env.local (см. ниже)
docker compose build
docker compose up -d
```

Проверка с сервера:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/
```

Логи:

```bash
docker compose logs -f app
```

---

## 5. Переменные окружения Guard (файл `guard-main/.env.local`)

Пример (подставьте свои значения; при связке через Docker-сеть URL ML уже задан в `docker-compose.yml`):

```env
# Публичный URL приложения (для ссылок в PDF/QR)
NEXT_PUBLIC_APP_URL=https://your-domain.example

# Секрет для QR-доступа к отчётам (≥ 16 символов в проде)
REPORT_ACCESS_SECRET=замените_на_длинную_случайную_строку

# Если URL ML не прописан в docker-compose environment:
# ANALYSIS_SERVICE_URL=http://analysis:8765
# ANALYSIS_SERVICE_API_KEY=если включён ключ на ML-сервисе
```

После изменения `.env.local` перезапустите контейнер `app`:

```bash
cd /opt/antiplag/repo/antiplagiarism/guard-main
docker compose up -d --force-recreate app
```

---

## 6. Проверка взаимодействия Guard ↔ ML

1. Откройте веб-интерфейс, выполните проверку документа на странице проверки.
2. Если связь есть, в результате отображается блок **«Расширенный анализ (Python / Qdrant)»** (метрики из ML).
3. На сервере в логах контейнера `app` не должно быть постоянных ошибок `[analysis-client]` при доступном ML.

Точечная проверка ML без браузера:

```bash
curl -sS http://127.0.0.1:8765/health
```

---

## 7. Порядок запуска после обновления кода

```bash
# ML
cd /opt/antiplag/repo/antiplagiarism
docker compose -f docker-compose.ml.yml up -d --build

# Guard
cd /opt/antiplag/repo/antiplagiarism/guard-main
docker compose up -d --build
```

---

## Альтернатива (проще): один `docker compose` на всё

Если хочешь разворачивать «как Guard» (одна команда и всё поднялось/связалось), используй единый файл:

```bash
cd /opt/antiplag/repo/antiplagiarism
docker compose -f docker-compose.server.yml up -d --build
```

Он поднимает сразу:

- `guard-app` + `guard-nginx`
- `analysis` + `qdrant`

И сразу включает связь **Guard → ML** через `ANALYSIS_SERVICE_URL=http://analysis:8765`.

Доступ: `http://<server-ip>:3000/login` (HTTPS появится только если добавишь SSL в nginx).

---

## 8. Если ML недоступен

Guard продолжит работать **только с MinHash** (без блока расширенного анализа), если **`ANALYSIS_SERVICE_URL`** не задан или сервис не отвечает.

---

## Справка по путям файлов

| Файл | Назначение |
|------|------------|
| `antiplagiarism/docker-compose.ml.yml` | Qdrant + контейнер `analysis` |
| `antiplagiarism/Dockerfile.analysis` | образ Python API |
| `antiplagiarism/INTEGRATION.md` | контракт API, env, Docker-сценарии |
| `guard-main/docker-compose.yml` | Guard + nginx |
