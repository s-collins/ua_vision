CREATE TABLE IF NOT EXISTS TrainingExample (
    id INTEGER NOT NULL,
    image_filepath CHAR(100),
    camera_angle INTEGER,
    camera_height INTEGER,
    light_angle INTEGER,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS Label (
    id INTEGER NOT NULL,
    image_id INTEGER,
    x1 INTEGER,
    x2 INTEGER,
    y1 INTEGER,
    y2 INTEGER,
    PRIMARY KEY (id),
    FOREIGN KEY (image_id) REFERENCES TrainingExample(id)
);
