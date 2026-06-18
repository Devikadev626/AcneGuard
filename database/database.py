import sqlite3
import os

from src.auth import hash_password, verify_password

# ===================================
# DATABASE PATH
# ===================================

DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "acneguard.db"
    )
)


# ===================================
# CONNECTION
# ===================================

def get_db_connection():
    """
    Creates and returns a SQLite connection.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ===================================
# DATABASE INITIALIZATION
# ===================================

def init_db():
    """
    Creates all required tables.
    """

    schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL CHECK(
            role IN (
                'Patient',
                'Doctor',
                'Technician',
                'Admin'
            )
        ),
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_name TEXT NOT NULL,
        severity TEXT NOT NULL CHECK(
            severity IN (
                'Grade 0',
                'Grade 1',
                'Grade 2',
                'Grade 3'
            )
        ),
        confidence REAL NOT NULL CHECK(
            confidence >= 0.0
            AND confidence <= 100.0
        ),
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id)
            REFERENCES users(id)
            ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL UNIQUE,
        report_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id)
            REFERENCES predictions(id)
            ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_predictions_user
    ON predictions(user_id);

    CREATE INDEX IF NOT EXISTS idx_reports_prediction
    ON reports(prediction_id);
    """

    with get_db_connection() as conn:
        conn.executescript(schema)
        conn.commit()


# ===================================
# USER FUNCTIONS
# ===================================

def create_user(
    username: str,
    email: str,
    role: str,
    password_hash: str
) -> int:
    """
    Creates a user record.
    """

    query = """
    INSERT INTO users (
        username,
        email,
        role,
        password_hash
    )
    VALUES (?, ?, ?, ?);
    """

    with get_db_connection() as conn:

        cursor = conn.execute(
            query,
            (
                username,
                email,
                role,
                password_hash
            )
        )

        conn.commit()

        return cursor.lastrowid


def register_user(
    username: str,
    email: str,
    role: str,
    password: str
) -> int:
    """
    Registers a user with a hashed password.
    """

    password_hash = hash_password(password)

    return create_user(
        username=username,
        email=email,
        role=role,
        password_hash=password_hash
    )


def get_user_by_username(username: str):
    """
    Returns user details by username.
    """

    query = """
    SELECT *
    FROM users
    WHERE username = ?;
    """

    with get_db_connection() as conn:

        row = conn.execute(
            query,
            (username,)
        ).fetchone()

        if row:
            return dict(row)

        return None


def authenticate_user(
    username: str,
    password: str
):
    """
    Validates username and password.
    """

    user = get_user_by_username(username)

    if not user:
        return None

    if verify_password(
        password,
        user["password_hash"]
    ):
        return user

    return None


def list_users():
    """
    Returns all users.
    """

    query = """
    SELECT
        id,
        username,
        email,
        role,
        created_at
    FROM users
    ORDER BY id DESC;
    """

    with get_db_connection() as conn:

        rows = conn.execute(query).fetchall()

        return [dict(row) for row in rows]


# ===================================
# PREDICTIONS
# ===================================

def create_prediction(
    image_name: str,
    severity: str,
    confidence: float,
    user_id: int = None
) -> int:
    """
    Creates prediction record.
    """

    query = """
    INSERT INTO predictions (
        image_name,
        severity,
        confidence,
        user_id
    )
    VALUES (?, ?, ?, ?);
    """

    with get_db_connection() as conn:

        cursor = conn.execute(
            query,
            (
                image_name,
                severity,
                confidence,
                user_id
            )
        )

        conn.commit()

        return cursor.lastrowid


def list_predictions(limit: int = 10):
    """
    Returns recent predictions.
    """

    query = """
    SELECT
        id,
        image_name,
        severity,
        confidence,
        prediction_date,
        user_id
    FROM predictions
    ORDER BY prediction_date DESC, id DESC
    LIMIT ?;
    """

    with get_db_connection() as conn:

        rows = conn.execute(
            query,
            (limit,)
        ).fetchall()

        return [dict(row) for row in rows]


def get_user_predictions(
    user_id: int,
    limit: int = 20
):
    """
    Returns predictions for a specific user.
    """

    query = """
    SELECT *
    FROM predictions
    WHERE user_id = ?
    ORDER BY prediction_date DESC
    LIMIT ?;
    """

    with get_db_connection() as conn:

        rows = conn.execute(
            query,
            (
                user_id,
                limit
            )
        ).fetchall()

        return [dict(row) for row in rows]


# ===================================
# REPORTS
# ===================================

def create_report(
    prediction_id: int,
    report_path: str
) -> int:
    """
    Maps report to prediction.
    """

    query = """
    INSERT INTO reports (
        prediction_id,
        report_path
    )
    VALUES (?, ?);
    """

    with get_db_connection() as conn:

        cursor = conn.execute(
            query,
            (
                prediction_id,
                report_path
            )
        )

        conn.commit()

        return cursor.lastrowid


# ===================================
# ANALYTICS
# ===================================

def get_analytics_data():
    """
    Returns all predictions for dashboard analytics.
    """

    query = """
    SELECT
        id,
        image_name,
        severity,
        confidence,
        prediction_date,
        user_id
    FROM predictions
    ORDER BY prediction_date DESC, id DESC;
    """

    with get_db_connection() as conn:

        rows = conn.execute(query).fetchall()

        return [dict(row) for row in rows]


# ===================================
# AUTO INIT
# ===================================

init_db()