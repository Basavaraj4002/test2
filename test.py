import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main import app
import json

def test_api():
    print("Creating test client...")
    client = TestClient(app)

    print("Generating a test image (a simple circle)...")
    # Create a dummy image (a white canvas with a black circle)
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    # Draw a clock circle
    cv2.circle(img, (250, 250), 200, (0, 0, 0), 4)
    
    # Let's add some numbers roughly (so OpenCV detects something)
    # We will just draw small circles where numbers should be
    for hour in range(1, 13):
        angle_deg = (hour * 30) - 90
        angle_rad = np.deg2rad(angle_deg)
        x = int(250 + 200 * 0.82 * np.cos(angle_rad))
        y = int(250 + 200 * 0.82 * np.sin(angle_rad))
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)

    # Draw clock hands (Minute pointing to 2, Hour pointing to 10)
    cv2.line(img, (250, 250), (250 + int(140 * np.cos(np.deg2rad(-30))), 250 + int(140 * np.sin(np.deg2rad(-30)))), (0, 0, 0), 4) # Minute hand -> 2
    cv2.line(img, (250, 250), (250 + int(90 * np.cos(np.deg2rad(-150))), 250 + int(90 * np.sin(np.deg2rad(-150)))), (0, 0, 0), 4) # Hour hand -> 10

    # Encode to base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Prepare the payload
    payload = {
        "patient_id": "test_patient_001",
        "age": 75,
        "education_years": 12,
        "image_base64": img_base64,
        "time_taken_seconds": 45,
        "stroke_count": 15,
        "hesitation_pauses": 2,
        "target_time": "10:10"
    }

    print("Sending POST request to /api/test/clock-drawing/submit...")
    response = client.post("/api/test/clock-drawing/submit", json=payload)

    print(f"\nStatus Code: {response.status_code}")
    if response.status_code == 200:
        print("\nResponse JSON:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error:")
        print(response.text)

if __name__ == "__main__":
    test_api()
