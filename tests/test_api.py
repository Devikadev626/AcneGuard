import os
import sys
from fastapi.testclient import TestClient

# Ensure project root is in path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.api import app
from database.database import list_predictions

client = TestClient(app)

def test_home_endpoint():
    print("Testing GET / endpoint...")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AcneGuard API Running"}
    print("GET / endpoint verified.")

def test_predict_endpoint_success():
    print("Testing POST /predict endpoint...")
    
    # Create test assets path
    test_image_dir = os.path.join("tests", "test_assets")
    os.makedirs(test_image_dir, exist_ok=True)
    test_image_path = os.path.join(test_image_dir, "test_acne.jpg")
    
    # Try to copy sample test image
    src_sample = "test_images/acne_sample.jpg"
    if os.path.exists(src_sample):
        import shutil
        shutil.copy(src_sample, test_image_path)
    else:
        # Create a simple red 224x224 placeholder image if sample is not found
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="red")
        img.save(test_image_path)

    # Retrieve initial database predictions count
    initial_predictions = list_predictions(limit=1000)
    initial_count = len(initial_predictions)

    # Perform POST /predict request
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test_acne.jpg", f, "image/jpeg")}
        )
    
    # Verify response status and structure
    assert response.status_code == 200, f"Request failed: {response.text}"
    json_resp = response.json()
    print(f"API Response: {json_resp}")
    
    assert "prediction_id" in json_resp, "Response must include prediction_id"
    assert "severity" in json_resp, "Response must include severity"
    assert "confidence" in json_resp, "Response must include confidence"
    
    # Clean up test image asset
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

    # Query database to confirm new record is saved
    updated_predictions = list_predictions(limit=1000)
    assert len(updated_predictions) == initial_count + 1, "Database record was not created!"
    
    latest_pred = updated_predictions[0]
    assert latest_pred["id"] == json_resp["prediction_id"], "DB prediction ID mismatch"
    assert latest_pred["image_name"] == "test_acne.jpg", "DB image name mismatch"
    assert latest_pred["severity"] == json_resp["severity"], "DB severity mismatch"
    assert latest_pred["confidence"] == json_resp["confidence"], "DB confidence mismatch"

    print("POST /predict integration and database storage verified successfully!")

if __name__ == "__main__":
    # Run test endpoints
    test_home_endpoint()
    print("-" * 50)
    test_predict_endpoint_success()
    print("-" * 50)
    print("All FastAPI integration tests PASSED successfully!")
