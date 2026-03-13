# GameLens Event Classifier

FastAPI service that classifies gameplay captures from PostgreSQL into event intervals.

## 📦 What's in this repo

- 🌐 FastAPI HTTP API for classifying gameplay captures.
- 🧠 PyTorch + timm inference pipeline for frame-level event classification.
- 🧩 Event segmentation module that turns frame labels into interval-based events for further data analytics.
- 🗄️ PostgreSQL integration via `psycopg` connection pooling.
- 🐳 Docker + Compose workflows for local and containerized runs.

## ✅ Requirements

- 🐍 Python `3.13+`
- ⚡ [`uv`](https://docs.astral.sh/uv/)
- 🗃️ A running PostgreSQL database
- 🐳 Docker + Docker Compose

## 🛠️ Installation
> [!IMPORTANT]  
> **Environment Variables:** You must configure the following in your `.env` file: 
> 
> For your database connection (`PGSQL_CONN`), the host depends on how you are running the API:
> * **Running via Docker:** `postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@db:5432/<POSTGRES_DB>`
>
>* This is true for docker when running both the DB container and Event Classifier Service in the same local enviornment.
> * **Running locally (Host machine):** `postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@localhost:5432/<POSTGRES_DB>`
>
> For your LLM api key, use: `OPENAI_API_KEY`.
> For a local build you also need to export those environment variables into the terminal.

## 🐳 Build for Docker Compose (Recommended)

1. Create the external network (one-time):

```bash
docker network create db_network
```

2. Build and run:

```bash
docker compose build
docker compose up -d
```

3. Logs and shutdown:

```bash
docker compose logs -f event_classifier
docker compose down
```

## 💻 Build for local development

Install dependencies:

```bash
uv sync
```

Run the app:

```bash
uv run fastapi dev main.py --host 0.0.0.0 --port 7761
```
