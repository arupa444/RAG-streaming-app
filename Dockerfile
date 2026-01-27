FROM python:3.11-slim

# 1. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 2. Copy requirements first (for caching)
COPY requirements.txt .

# 3. Install dependencies
RUN uv pip install --system --no-cache -r requirements.txt

# 4. Copy the rest of the app
#    (This includes .streamlit/config.toml, but EXCLUDES secrets.toml due to .dockerignore)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]