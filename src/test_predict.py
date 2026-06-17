from predict import predict_image

severity, confidence = predict_image(
    "test.jpg"
)

print(
    f"Severity: {severity}"
)

print(
    f"Confidence: {confidence}%"
)