

create a virtual environment :- 
python -m venv venv

source venv/bin/activate



1.) First Create Pipeline/data_pipeline.py 

# to run the app use: 
uvicorn app:app 

# To build the docker image 

```bash 
docker build -t house-price-api . 
```

# To run the docker container 
```bash
docker run -p 8000:8000 house-price-api 

```



























