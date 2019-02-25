#!/bin/bash

# directory containing the pipeline
PIPELINE=$1

# Build the Docker image
docker build -t $PIPELINE $PIPELINE

# Run the docker container
#docker run --name $PIPELINE --rm -it $PIPELINE
docker run --name $PIPELINE --rm -it $PIPELINE