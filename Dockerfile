FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn python-multipart torch torchvision pillow

COPY src/sps_genai /app/sps_genai
COPY cnn_cifar10.pth /app/cnn_cifar10.pth

EXPOSE 8000

CMD ["uvicorn", "sps_genai.image_api:app", "--host", "0.0.0.0", "--port", "8000"]
