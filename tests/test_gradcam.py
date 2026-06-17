import os
import sys
import torch
import torch.nn as nn
from torchvision import models

# Ensure project root is in path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.gradcam import generate_gradcam_overlay

def test_gradcam_execution():
    print("Testing Grad-CAM implementation...")
    
    # Define verification paths
    model_path = "models/best_acne_model.pth"
    test_image_path = "test_images/acne_sample.jpg"
    output_overlay_path = "tests/test_assets/gradcam_output.png"
    
    assert os.path.exists(model_path), f"Model weights file not found at: {model_path}"
    assert os.path.exists(test_image_path), f"Sample image file not found at: {test_image_path}"
    
    # Instantiate and load model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ResNet-18 model checkpoint on: {device}")
    
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 4)
    )
    
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    
    # Establish target layer for feature map attention (ResNet-18 Layer 4)
    target_layer = model.layer4
    
    # Remove previous test output if it exists
    if os.path.exists(output_overlay_path):
        try:
            os.remove(output_overlay_path)
        except Exception as e:
            print(f"Failed to clear old test assets: {e}")
            
    # Generate visual explanation overlay using Grad-CAM
    print(f"Executing Grad-CAM overlay generation for image: {test_image_path}")
    result_path = generate_gradcam_overlay(
        model=model,
        target_layer=target_layer,
        image_path=test_image_path,
        output_path=output_overlay_path,
        alpha=0.4
    )
    
    # Assert output exists and is populated
    assert os.path.exists(result_path), "Visual overlay image was not created!"
    assert os.path.getsize(result_path) > 0, "Visual overlay image is empty (0 bytes)!"
    print(f"Verified: Grad-CAM overlay image generated successfully at '{result_path}' (Size: {os.path.getsize(result_path)} bytes).")
    
    # Clean up test output asset
    if os.path.exists(output_overlay_path):
        os.remove(output_overlay_path)
        print("Temporary test assets cleaned up successfully.")
        
    print("Grad-CAM module tests PASSED successfully!")

if __name__ == "__main__":
    test_gradcam_execution()
