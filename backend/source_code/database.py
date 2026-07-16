import sqlite3
import os

# DB will be created in the backend root
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            predicted_weight_g REAL NOT NULL,
            length_cm REAL,
            thickness_cm REAL,
            perimeter_cm REAL,
            area_cm2 REAL,
            volume_proxy_cm3 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
