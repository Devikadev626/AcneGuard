import os
import sys

# Ensure project root is in path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from database.database import (
    init_db,
    create_user,
    create_prediction,
    list_predictions,
    create_report,
    DB_PATH
)

def test_database_lifecycle():
    # 1. Clean up old test database file if exists
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print("Removed existing test database file.")
        except Exception as e:
            print(f"Could not remove DB file: {e}")

    # 2. Initialize database
    print("Initializing SQLite Database...")
    init_db()
    assert os.path.exists(DB_PATH), "Database file was not created!"
    print("Database file verified successfully.")

    # 3. Create a test user
    print("Creating test user...")
    user_id = create_user(
        username="john_doe",
        email="john.doe@example.com",
        role="Patient"
    )
    print(f"User created with ID: {user_id}")
    assert user_id == 1, "User ID should be 1."

    # 4. Create prediction linked to user
    print("Creating prediction with user...")
    pred1_id = create_prediction(
        image_name="acne_severe.jpg",
        severity="Grade 3",
        confidence=94.50,
        user_id=user_id
    )
    print(f"Prediction created with ID: {pred1_id}")
    assert pred1_id == 1, "Prediction ID should be 1."

    # 5. Create prediction without user link (anonymous prediction)
    print("Creating prediction without user...")
    pred2_id = create_prediction(
        image_name="acne_mild.jpg",
        severity="Grade 1",
        confidence=82.15
    )
    print(f"Prediction created with ID: {pred2_id}")
    assert pred2_id == 2, "Prediction ID should be 2."

    # 6. List predictions
    print("Fetching prediction list...")
    predictions = list_predictions(limit=5)
    print(f"Fetched {len(predictions)} predictions:")
    for pred in predictions:
        print(f" - ID: {pred['id']}, Image: {pred['image_name']}, Severity: {pred['severity']}, Confidence: {pred['confidence']}%, User ID: {pred['user_id']}")
    
    assert len(predictions) == 2, "Should return 2 prediction entries."
    # The newest should be first (LIFO order by date)
    assert predictions[0]['id'] == 2, "First element in predictions list should be ID 2 (newest)."

    # 7. Create report linked to prediction
    print("Creating report mapping...")
    report_id = create_report(
        prediction_id=pred1_id,
        report_path="reports/acne_report_1.pdf"
    )
    print(f"Report mapping created with ID: {report_id}")
    assert report_id == 1, "Report ID should be 1."

    print("\nAll database integration tests PASSED successfully!")

if __name__ == "__main__":
    test_database_lifecycle()
