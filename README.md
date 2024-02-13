# mlservice

Example of an on-demand, offline training service that trains a simple ML model and performs inference through a REST-API

## Inference API

### Running script
python3 -m venv mlservice-env
source mlservice-env/bin/activate
cd inference
pip3 install -t requirements.txt
./run.sh

### Running Docker 

mlservice/inference╱$ docker build --tag inference-api .

docker run -it -p 5000:5000 inference-api

DEBUGGING: docker run -it --entrypoint /bin/bash inference-api

sharing a docker: 
docker save --output latestversion-1.0.0.tar dockerregistry/latestversion:1.0.0
docker load --input latestversion-1.0.0.tar

