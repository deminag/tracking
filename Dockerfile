FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    python3-pip \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p static/uploads static/results

EXPOSE 5000

ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "app.py"]