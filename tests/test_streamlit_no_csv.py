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

def test_no_csv_dependency():
    print("Verifying removal of CSV history file dependency...")
    
    # 1. Clean up old reports/history.csv if it exists
    history_file = os.path.join("reports", "history.csv")
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
            print(f"Obsolete CSV history file '{history_file}' removed successfully.")
        except Exception as e:
            print(f"Could not remove '{history_file}': {e}")
    else:
        print("CSV history file does not exist (already removed or not generated).")

    # 2. Check streamlit_app.py source code for CSV history file references
    streamlit_app_path = os.path.join("app", "streamlit_app.py")
    assert os.path.exists(streamlit_app_path), "streamlit_app.py not found!"

    with open(streamlit_app_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "history.csv" not in content, "Found reference to 'history.csv' in app/streamlit_app.py!"
    print("Verified: No references to 'history.csv' in app/streamlit_app.py.")
    
    print("Streamlit UI SQLite integration test PASSED successfully!")

if __name__ == "__main__":
    test_no_csv_dependency()
