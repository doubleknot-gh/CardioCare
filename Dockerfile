FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/sample_input.csv ./data/sample_input.csv

CMD ["python", "src/inference.py", "--input", "data/sample_input.csv"]