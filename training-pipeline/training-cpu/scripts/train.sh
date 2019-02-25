#!/bin/bash
#-------------------------------------------------------------------------------
# Preconditions:
#  - Training pipeline has been saved in the root directory
#-------------------------------------------------------------------------------

# Training Data
DATASET="dataset1"
DATA_URL="https://github.com/s-collins/ua_vision_data/archive/${DATASET}.tar.gz"

# Base model
MODEL_CONFIG_PATH="/pipeline/base_models/ssd_mobilenet_v1_pets.config"
MODEL_CHECKPOINT="ssd_mobilenet_v1_coco_2018_01_28"
MODEL_CHECKPOINT_URL="http://download.tensorflow.org/models/object_detection/${MODEL_CHECKPOINT}.tar.gz"

# Download and expand the raw training data inside the root directory
cd /
wget $DATA_URL && tar xvsf ${DATASET}.tar.gz

# Download training checkpoint
cd /
wget $MODEL_CHECKPOINT_URL && tar xvsf ${MODEL_CHECKPOINT}.tar.gz

# Partition dataset into training and evaluation subsets
python /pipeline/scripts/generate_input.py \
  --training_output=/pipeline/input/data/test.tfrecords \
  --evaluation_output=/pipeline/input/data/eval.tfrecords

# Start the training
python /models/research/object_detection/legacy/train.py \
  --logtostderr \
  --train_dir=/pipeline/training \
  --pipeline_config_path=$MODEL_CONFIG_PATH
