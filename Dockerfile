FROM python:3.9-slim 

WORKDIR /app 

COPY requirements.txt .

# RUN pip install --no-cache-dir --upgrade pip \ 
#     && pip uninstall -y numpy \ 
#     && pip install --no-cache-dir numpy==1.26.2 \ 
#     && pip install --no-cache-dir -r requirements.txt 

RUN pip install --no-cache-dir --upgrade pip \
    && pip uninstall -y numpy \
    && pip install --no-cache-dir numpy==1.26.2 \
    && pip install --no-cache-dir -r requirements.txt

# COPY project files 
COPY . . 

# Expose FastAPI port 
EXPOSE 8000 

# Start FastAPI server 
CMD ["uvicorn", "app:app"] 







