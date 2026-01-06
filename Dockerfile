FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install -U pip && pip install -e ".[dev]"

ENV STOCK_SCREENER_CACHE_DIR=/data \
    STOCK_SCREENER_DATA_BACKEND=sqlite

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "stock_screener.server:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]

