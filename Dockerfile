# Python 3.12 — broad, stable wheel support for torch/opencv/scipy
FROM python:3.12-slim

# System libs that OpenCV/torch need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as a non-root user with UID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install Python deps first (better build caching). torch comes in via ultralytics.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project (includes scanner/, the .pt models, etc.)
COPY --chown=user . .

USER user

# Collect static at build (no DB needed here; uses the dev SECRET_KEY fallback)
RUN python Infraestimator/manage.py collectstatic --noinput

# HF Spaces expects the app on port 7860
EXPOSE 7860

# At startup: run migrations (DB secrets are available now), then launch gunicorn.
# --timeout 120 so a slow first inference doesn't get killed.
CMD python Infraestimator/manage.py migrate --noinput && \
    gunicorn --chdir Infraestimator --bind 0.0.0.0:7860 --timeout 120 Infraestimator.wsgi:application