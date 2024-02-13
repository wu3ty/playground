FROM tensorflow/tensorflow

# Install Linux dependencies
RUN apt-get update 
RUN apt-get -y install python3 python3-pip

# Install Python requirements
COPY ./requirements.txt .
RUN pip install --ignore-installed -r requirements.txt

# Create a working directory and copy application into container
WORKDIR /app
COPY ./main.py .

# Start API
ENTRYPOINT ["python3", "main.py"]
