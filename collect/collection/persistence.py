import sqlite3
import cv2 as cv
import yaml
import os

# ------------------------------------------------------------------------------
# Module initialization
# ------------------------------------------------------------------------------

# Load settings
stream = open('settings.yaml', 'r')
settings = yaml.load(stream)
stream.close()

# Construct filepaths
context = os.path.abspath('../..') + '/'
images_dir = settings['root_dir'] + settings['images_dir']
db_dir = settings['root_dir'] + settings['db_dir']
db_path = db_dir + settings['db_name']

# Create directories if they don't already exist
if not os.path.exists(context + settings['root_dir']):
	os.makedirs(context + images_dir)
	os.makedirs(context + db_dir)

# Generate schema if does not already exist
schema_definition = open('schema.sql', 'r').read()
conn = sqlite3.connect(context + db_path)
cursor = conn.cursor()
cursor.executescript(schema_definition)
conn.commit()
conn.close()

# ------------------------------------------------------------------------------
# Persistence class
# ------------------------------------------------------------------------------

class Persistence:

	def __init__(self):
		"""Connects to database"""
		self.conn = sqlite3.connect(context + db_path)

	def __del__(self):
		"""Disconnects from database"""
		self.conn.close()

	def save(self, ex):
		"""Persists training example image and metadata."""

		# Save metadata (except for image path, which depends on record id)
		c = self.conn.cursor()
		query = """\
		INSERT INTO TrainingExample (camera_angle, camera_height, light_angle)
		VALUES (?,?,?)
		"""
		c.execute(query, (ex.camera_angle, ex.camera_height, ex.light_angle))
		ex_id = c.lastrowid

		# Save image
		image_path = images_dir + str(ex_id) + '.jpg'
		cv.imwrite(context + image_path, ex.image)

		# Save image path (now that we know record id)
		query = """\
		UPDATE TrainingExample
		SET image_filepath = ?
		WHERE id = ?
		"""
		c.execute(query, (image_path, ex_id))

		# Save labels
		query = """\
		INSERT INTO LABEL (image_id, x1, x2, y1, y2)
		VALUES (?, ?, ?, ?, ?)
		"""
		for label in ex.labels:
			x1, x2, y1, y2 = label[0][0], label[1][0], label[0][1], label[1][1]
			c.execute(query, (ex_id, x1, x2, y1, y2))

		# Commit changes
		self.conn.commit()

