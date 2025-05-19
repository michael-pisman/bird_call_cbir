FROM python:3.13-slim

# system deps for librosa/soundfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy the code
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# default entrypoint runs your pipeline
ENTRYPOINT ["python", "main.py"]