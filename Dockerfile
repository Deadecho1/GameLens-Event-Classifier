FROM python:latest

# Copy the uv binary directly from the official astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./

RUN uv sync

COPY . .

EXPOSE 7761

CMD ["uv","run","fastapi","dev","main.py","--host","0.0.0.0","--port","7761"]
