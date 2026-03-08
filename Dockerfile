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
ENV PYTHONUNBUFFERED=1

# NOTE: Non-root user disabled — Railway mounts /data volume at runtime,
# overwriting image-layer chown. The quant user would lose write access to
# /data/state.db and all persistent files. Re-enable only when Railway adds
# volume-mount UID configuration or a startup chown entrypoint is added.
# RUN useradd -m -u 1001 quant && mkdir -p /data && chown quant:quant /data
# USER quant

CMD ["python", "main.py"]
