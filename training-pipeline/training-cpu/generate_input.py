import tensorflow as tf
from object_detection.utils import dataset_util
import random
import shutil
import sqlite3
import os
import cv2 as cv
import base64

flags = tf.app.flags
flags.DEFINE_string('training_output', 'input/data/test.tfrecords', 'Path to training output')
flags.DEFINE_string('evaluation_output', 'input/data/eval.tfrecords', 'Path to training output')
FLAGS = flags.FLAGS


def create_tf_examples(ids, images_dir, db_path):
	"""Generates a TFRecord for given set of training examples.

	Inputs:
		ids         - list of training example IDs
		images_dir  - absolute path to directory containing images
		db_path     - absolute path to database with metadata and labels

	"""

	# list of training examples
	examples = []

	# open the database
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	# populate the list of training examples
	for id in ids:
		# get image dimensions
		img_path = images_dir + '/' + str(id) + '.jpg'
		img = cv.imread(img_path, cv.IMREAD_COLOR)
		height, width, channels = img.shape

		# populate dictionary with training example data
		cursor.execute("SELECT * FROM TrainingExample WHERE id=?", (id,))
		row = cursor.fetchone()
		example = {
			'id': row[0],
			'img_path': img_path,
			'img_height': height,
			'img_width': width,
			'camera_angle': row[2],
			'camera_height': row[3],
			'light_angle': row[4],
			'labels': []
		}

		# populate label information
		cursor.execute("SELECT * FROM Label WHERE image_id=?", (id,))
		for row in cursor.fetchall():
			example['labels'].append({
				'id': row[0],
				'img_id': row[1],
				'xmin': min(row[2], row[3]),
				'xmax': max(row[2], row[3]),
				'ymin': min(row[4], row[5]),
				'ymax': max(row[4], row[5])
			})

		examples.append(example)

	# close the database
	cursor.close()
	conn.close()

	# Create TFRecord
	tf_examples = []
	for example in examples:
		# encode image bytes
		encoded_image_data = tf.gfile.FastGFile(example['img_path'], 'rb').read()

		if len(example['labels']) != 1:
			continue

		# normalize label coordinates
		xmins = []
		xmaxs = []
		ymins = []
		ymaxs = []
		classes_text = []
		classes = []
		for label in example['labels']:
			xmins.append(label['xmin'] / example['img_width'])
			xmaxs.append(label['xmax'] / example['img_width'])
			ymins.append(label['ymin'] / example['img_height'])
			ymaxs.append(label['ymax'] / example['img_height'])
			classes_text.append(b'rock')
			classes.append(1)

		# create class text to class id mapping
		#classes_text = [b'rock']
		#classes = [1]

		# populate a TFRecord entry for the training example
		tf_examples.append(tf.train.Example(features=tf.train.Features(feature={
	        'image/height': dataset_util.int64_feature(example['img_height']),
	        'image/width': dataset_util.int64_feature(example['img_width']),
	        'image/filename': dataset_util.bytes_feature(example['img_path'].encode('utf8')),
	        'image/source_id': dataset_util.bytes_feature(example['img_path'].encode('utf8')),
	        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
	        'image/format': dataset_util.bytes_feature(b'jpeg'),
	        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
	        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
	        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
	        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
	        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
	        'image/object/class/label': dataset_util.int64_list_feature(classes),
		})))

	# Return the TFRecord entries
	return tf_examples


def main(_):
	# Detect number of training examples
	images_path = '/data/images'
	num_examples = len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path,name))])

	# Randomly subdivide examples into training and evaluation sets
	# TODO: In the future, this setting should be loaded when the Docker container is executed.
	training_ratio = 0.8
	random.seed()
	training_ids = random.sample(range(1, num_examples + 1), int(training_ratio * num_examples))
	eval_ids = list(set(range(1, num_examples + 1)) - set(training_ids))

	# Create directories to contain training input
	os.mkdir('input')
	os.mkdir('input/training')
	os.mkdir('input/data')
	os.mkdir('input/images')
	os.mkdir('input/images/test')
	os.mkdir('input/images/eval')

	# Put images in "input/images/test" and "input/images/eval" directories
	for id in training_ids:
		src = images_path + '/' + str(id) + '.jpg'
		dest = 'input/images/test/' + str(id) + '.jpg'
		shutil.copyfile(src, dest)
	for id in eval_ids:
		src = images_path + '/' + str(id) + '.jpg'
		dest = 'input/images/eval/' + str(id) + '.jpg'
		shutil.copyfile(src, dest)

	# Generate TFRecord for training examples
	writer = tf.python_io.TFRecordWriter(FLAGS.training_output)
	examples = create_tf_examples(training_ids, 'input/images/test', 'data/database/training_examples.sqlite3')
	for e in examples:
		writer.write(e.SerializeToString())
	writer.close()

	# Generate TFRecord for evaluation examples
	writer = tf.python_io.TFRecordWriter(FLAGS.evaluation_output)
	examples = create_tf_examples(eval_ids, 'input/images/eval', 'data/database/training_examples.sqlite3')
	for e in examples:
		writer.write(e.SerializeToString())
	writer.close()

	# Put the model configuration file and label map into "input/training" directory
	shutil.copytree('/base_models', '/training-cpu/input/base_models')


if __name__ == '__main__':
	tf.app.run()
	print('Done')