FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Keep dependency installation cached unless the requirements change.
COPY requirments.txt ./requirments.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirments.txt

COPY app.py ./app.py
COPY src/ ./src/

# The app writes its SQLite checkpointer relative to its working directory.
RUN mkdir -p /data \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /data

WORKDIR /data
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
