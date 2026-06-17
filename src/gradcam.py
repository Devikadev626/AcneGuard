import cv2
import numpy as np
import torch
import os
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes hooks on the specified target layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """
        Removes the registered hooks to avoid keeping reference in the computational graph.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Runs backward pass to compute Grad-CAM heatmap.
        Returns the raw 2D normalized heatmap and the predicted/target class index.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass for target class score
        score = output[0, class_idx]
        score.backward()
        
        # Retrieve gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]      # Shape: (C, H, W)
        activations = self.activations.cpu().data.numpy()[0]  # Shape: (C, H, W)
        
        # Global Average Pooling of gradients (weights)
        weights = np.mean(gradients, axis=(1, 2))  # Shape: (C,)
        
        # Linear combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to filter out features that do not positively influence the score
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1] range
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, class_idx

def generate_gradcam_overlay(model, target_layer, image_path, output_path, alpha=0.4):
    """
    Computes the Grad-CAM heatmap, applies cv2.COLORMAP_JET, overlays it on the 
    original image using OpenCV blending, and saves it to output_path.
    """
    # 1. Load original image and determine dimensions
    original_img = Image.open(image_path).convert("RGB")
    width, height = original_img.size
    
    # 2. Preprocess input tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Get computation device
    device = next(model.parameters()).device
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # 3. Generate raw Grad-CAM heatmap
    cam_extractor = GradCAM(model, target_layer)
    try:
        heatmap, class_idx = cam_extractor.generate_heatmap(input_tensor)
    finally:
        # Guarantee hook removal
        cam_extractor.remove_hooks()
        
    # 4. Bilinear resize heatmap back to original image size
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Convert heatmap to uint8 range [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # 5. Apply OpenCV JET colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 6. Convert BGR (OpenCV default) to RGB
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 7. AddWeighted blend of original image and heatmap colormap
    original_arr = np.array(original_img)
    overlay_arr = cv2.addWeighted(original_arr, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    
    # 8. Convert to PIL Image and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_img = Image.fromarray(overlay_arr)
    overlay_img.save(output_path)
    
    return output_path
