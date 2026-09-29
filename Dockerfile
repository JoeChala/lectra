FROM python:3.12-slim

WORKDIR /app

COPY backend/pyproject.toml backend/uv.lock* ./

RUN pip install --no-cache-dir uv

RUN uv sync --frozen --no-dev

COPY backend/src ./src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "lectra.main:app", "--host", "0.0.0.0", "--port", "8000"]