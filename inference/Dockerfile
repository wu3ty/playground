# get pytorch base
FROM ghcr.io/pytorch/pytorch:2.0.1-runtime

# Install Python requirements
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Create a working directory and copy application into container
WORKDIR /app
COPY ./main.py .

# Start API
ENTRYPOINT ["python3", "main.py"]
