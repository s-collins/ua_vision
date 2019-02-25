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

# Pull Tensorflow models source code
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

# Copy the pipeline into the image
COPY . /pipeline

# Install wget
RUN apt-get install wget

ENTRYPOINT bash /pipeline/scripts/train.sh