FROM python:3.12.7

WORKDIR /app

COPY src/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY /src .

EXPOSE 5040

CMD ["python", "pnd_detection_model.py"]