FROM python:3.10

WORKDIR /app

COPY src/insurance_mlops/api/app.py ./app.py
COPY src/insurance_mlops/models/ ./models/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV ENV=production

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

