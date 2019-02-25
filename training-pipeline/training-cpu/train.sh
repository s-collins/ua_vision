#!/bin/bash

# Copy the data directory into cwd
cp -r ../../data .

# Copy the models directory into cwd
cp -r ../base_models .

# Build the Docker image
docker build -t training .

# Run the docker container
docker run --name training_test --rm -it training

# Clean-up
rm -rf data
rm -rf base_models