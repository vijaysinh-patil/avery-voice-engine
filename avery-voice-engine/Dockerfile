FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        espeak-ng \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py download_models.py ./

ENV MODEL_DIR=/data/models

EXPOSE 10200

CMD ["sh", "-c", "python download_models.py && python run.py --model /data/models/kokoro-v1.0.int8.onnx --voices /data/models/voices-v1.0.bin --uri tcp://0.0.0.0:10200"]
