FROM python:3.12.1

WORKDIR /app

COPY src/ ./src/

RUN pip install numpy==1.26.3 gym==0.26.2 matplotlib==3.8.3

CMD ["python", "src/main.py"]