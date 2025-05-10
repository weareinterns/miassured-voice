FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose HTTP port (for both health checks and WebSocket)
EXPOSE 8080

CMD ["python", "HealioVoice.py"] 
