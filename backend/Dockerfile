FROM python:3.9-slim-buster

WORKDIR /app

# Copy the Flask app code and requirements
COPY app.py .
COPY classifier.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the PyTorch model weight file and directories
COPY Net/ ./Net/
COPY PredictResult/ ./PredictResult/
COPY ReceivedFile/ ./ReceivedFile/

CMD flask run -h 0.0.0.0 -p 5000