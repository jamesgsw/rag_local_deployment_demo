FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY src/ .

RUN pip install --no-cache-dir -r requirements.txt

ENV TOKENIZERS_PARALLELISM="false"

CMD ["python", "main.py"]