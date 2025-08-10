FROM python:3.9-slim

WORKDIR /app

# 시스템 의존성 설치 (필요한 경우)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

EXPOSE 8000

CMD ["python", "api.py"]