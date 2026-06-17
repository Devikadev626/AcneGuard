import sqlite3
import os

# Define absolute path to the SQLite database file
DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "acneguard.db"
    )
)

def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.
    Enforces row_factory to sqlite3.Row for dict-like access,
    and enables foreign key constraint checks.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """
    Creates the required tables: users, predictions, and reports
    along with their standard structural relations, column definitions, and constraints.
    """
    schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('Patient', 'Doctor', 'Technician', 'Admin')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_name TEXT NOT NULL,
        severity TEXT NOT NULL CHECK(severity IN ('Grade 0', 'Grade 1', 'Grade 2', 'Grade 3')),
        confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 100.0),
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL UNIQUE,
        report_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
    CREATE INDEX IF NOT EXISTS idx_reports_prediction ON reports(prediction_id);
    """
    with get_db_connection() as conn:
        conn.executescript(schema)
        conn.commit()

def create_user(username: str, email: str, role: str) -> int:
    """
    Creates a new user record.
    Returns the auto-generated id of the user.
    """
    query = """
    INSERT INTO users (username, email, role)
    VALUES (?, ?, ?);
    """
    with get_db_connection() as conn:
        cursor = conn.execute(query, (username, email, role))
        conn.commit()
        return cursor.lastrowid

def create_prediction(image_name: str, severity: str, confidence: float, user_id: int = None) -> int:
    """
    Creates a new acne prediction entry.
    Returns the auto-generated id of the prediction.
    """
    query = """
    INSERT INTO predictions (image_name, severity, confidence, user_id)
    VALUES (?, ?, ?, ?);
    """
    with get_db_connection() as conn:
        cursor = conn.execute(query, (image_name, severity, confidence, user_id))
        conn.commit()
        return cursor.lastrowid

def list_predictions(limit: int = 10) -> list:
    """
    Returns the list of predictions, ordered from newest to oldest, up to `limit`.
    Each item is returned as a dictionary.
    """
    query = """
    SELECT id, image_name, severity, confidence, prediction_date, user_id
    FROM predictions
    ORDER BY prediction_date DESC, id DESC
    LIMIT ?;
    """
    with get_db_connection() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(row) for row in rows]

def create_report(prediction_id: int, report_path: str) -> int:
    """
    Maps a compiled report to a prediction record.
    Returns the auto-generated id of the report.
    """
    query = """
    INSERT INTO reports (prediction_id, report_path)
    VALUES (?, ?);
    """
    with get_db_connection() as conn:
        cursor = conn.execute(query, (prediction_id, report_path))
        conn.commit()
        return cursor.lastrowid
