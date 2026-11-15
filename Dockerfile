# Stage 1: Build isolated virtual environment with pinned wheels
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Hardened non-root runtime image
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --chown=app:app src ./src
COPY --chown=app:app streamlit_app.py ./streamlit_app.py
COPY --chown=app:app [".streamlit", "./.streamlit"]
COPY --chown=app:app ["ELHADJI Oussama", "./ELHADJI Oussama"]

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
