DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    image_filename TEXT
);