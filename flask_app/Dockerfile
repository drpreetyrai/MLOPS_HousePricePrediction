From python:3.9-slim

WORKDIR /app 

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt || echo "Dependency installation failed"

COPY . . 

EXPOSE 5000 

CMD ["python", "app.py"]







