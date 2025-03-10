# Sample commands to build docker image and run container
# docker build -t rag-app .
# docker run --rm -it rag-app

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY src/ ./src
COPY data /data

RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "src/main.py"]