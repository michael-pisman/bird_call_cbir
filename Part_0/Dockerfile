FROM python:3.12-slim

# system deps for librosa/soundfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy & install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the code
COPY . .

# default entrypoint runs your pipeline
ENTRYPOINT ["python", "main.py"]