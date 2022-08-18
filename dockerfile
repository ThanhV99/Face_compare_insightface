FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app/

CMD ["python", "server.py"]