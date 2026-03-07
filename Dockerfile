FROM python:3.12-slim-bookworm

WORKDIR /app

# Install deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY *.py ./
COPY templates ./templates
COPY alembic ./alembic
COPY alembic.ini ./
RUN test -f /app/templates/index.html || (echo "ERROR: templates/index.html missing" && exit 1)
COPY .env.example ./

# Persistent state volume — mount /data in Railway
ENV DATABASE_URL=sqlite+aiosqlite:////data/state.db

# Non-root user
RUN useradd -m -u 1001 quant && mkdir -p /data && chown quant:quant /data
USER quant

CMD ["python", "main.py"]
