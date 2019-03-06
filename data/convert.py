from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
import io
import os
from lxml import etree
import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


# Command-line flags
flags = tf.app.flags
flags.DEFINE_string('annotations_dir', 'annotations', 'Root directory to PASCAL annotations.')
flags.DEFINE_string('image_dir', 'images', 'Root directory to raw images.')
flags.DEFINE_string('label_map_path', 'label_map.pbtxt')
flags.DEFINE_string('output_path', 'examples.tfrecord', 'Path to output TFRecord.')


#TODO: Add "dict_to_tf_example"
def dict_to_tf_example(data, label_map_dict):
	# Verify file format
	img_path = os.path.join(FLAGS.image_dir, data['filename'])
	with tf.gfile.GFile(img_path, 'rb') as fid:
		encoded_jpg = fid.read()
	encoded_jpg_io = io.BytesIO(encoded_jpg)
	image = PIL.Image.open(encoded_jpg_io)
	if image.format != 'JPEG':
		raise ValueError('Image format is not JPEG')

	# Create key
	key = hashlib.sha256(encoded_jpg).hexdigest()

	width = int(data['size']['width'])
	height = int(data['size']['height'])

	# Create array of class labels for this training example
	xmin = []
	ymin = []
	xmax = []
	ymax = []
	classes = []
	classes_text = []
	if 'object' in data:
		for obj in data['object']:
			xmin.append(float(obj['bndbox']['xmin']) / width)
			ymin.append(float(obj['bndbox']['ymin']) / height)
			xmax.append(float(obj['bndbox']['xmax']) / width)
			ymax.append(float(obj['bndbox']['ymax']) / height)
			classes_text.append(obj['name'].encode('utf8'))
			classes.append(label_map_dict[obj['name']])

	# Create the training example
	example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
		'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
		'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
		'image/encoded': dataset_util.bytes_feature(encoded_jpg),
		'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
  		'image/object/class/label': dataset_util.int64_list_feature(classes)
	}))
	return example


def main(_):
	# Extract command-line flags
	annotations_dir = FLAGS.annotations_dir
	image_dir = FLAGS.image_dir
	label_map_path = FLAGS.label_map_path
	output_path = FLAGS.output_path

	# Object to write to TFRecord file
	writer = tf.python_io.TFRecordWriter(output_path)

	# Dictionary of class label ID to class label name
	label_map_dict = label_map_util.get_label_map_dict(label_map_path)

	num_examples = 270

	for i in range(1, num_examples + 1):
		# Load the XML file for training example
		path = os.path.join(annotations_dir, str(i) + '.xml')
		with tf.gfileGFile(path, 'r') as fid:
			xml_str = fid.read()
		xml = etree.fromstring(xml_str)
		data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

		# Convert dictionary to TFRecord object
		tf_example = dict_to_tf_example(data, label_map_dict):

		# Save the record
		writer.write(tf_example.SerializeToString())

	writer.close()

if __name__ == '__main__':
	tf.app.run()