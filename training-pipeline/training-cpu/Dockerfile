FROM ubuntu

# Install pip3 and git
RUN apt-get update \
    && apt-get install python-pip git-core -y

# Install Tensorflow (CPU)
RUN pip install tensorflow

# Install dependencies for Tensorflow object_detection API
RUN apt-get install protobuf-compiler python-pil python-lxml -y
RUN pip install Cython contextlib2 jupyter matplotlib

# Download Python3 setuptools
RUN apt-get install python-setuptools -y

# Pull Tensorflow models source code and run setup script
RUN git clone https://github.com/tensorflow/models.git

# Install COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && make \
    && cp -r pycocotools /models/research/

# Compile the Protobuf library
RUN cd models/research \
    && protoc object_detection/protos/*.proto --python_out=.

# Add tensorflow/models/research and slim dirs to PYTHONPATH
ENV PYTHONPATH /models/research/:/models/research/slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install python-opencv -y

# Copy the data directory into the image
COPY data /data
COPY . /training-cpu

# Copy the models into the image
COPY base_models /base_models

# Generate training and evaluation sets
WORKDIR /training-cpu
RUN python generate_input.py

# Start the training
WORKDIR /models/research/object_detection
RUN cp legacy/train.py . && cp legacy/eval.py .
RUN python train.py \
    --logtostderr \
    --train_dir=/training-cpu/input/training \
    --pipeline_config_path=/training-cpu/input/base_models/faster_rcnn_inception_v2_pets.config
