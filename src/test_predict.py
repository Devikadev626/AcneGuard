from predict import predict_image

severity, confidence = predict_image(
    "test_images/acne_sample.jpg"
)

print(
    f"Severity: {severity}"
)

print(
    f"Confidence: {confidence}%"
)