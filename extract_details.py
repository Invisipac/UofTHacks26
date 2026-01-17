import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time
# -----------------------------
# Download model if needed
# -----------------------------
model_path = 'face_landmarker.task'
if not os.path. exists(model_path):
    print("Downloading face landmarker model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker. task'
    urllib.request. urlretrieve(url, model_path)
    print("Model downloaded successfully!")

# -----------------------------
# MediaPipe Face Mesh setup (NEW API)
# -----------------------------
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create FaceLandmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# -----------------------------
# Helper functions
# -----------------------------
def extract_region(image, landmarks, indices):
    """Extract polygon region from facial landmarks"""
    h, w, _ = image.shape
    points = np.array([
        (int(landmarks[i]. x * w), int(landmarks[i].y * h))
        for i in indices
    ])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    region = cv2.bitwise_and(image, image, mask=mask)
    return region, mask


def analyze_skin(region):
    """Return oiliness, redness, texture metrics"""
    # Remove black background
    mask = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = region[mask > 0]

    if len(region) == 0:
        return None

    # Color spaces
    hsv = cv2.cvtColor(region. reshape(-1,1,3), cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(region.reshape(-1,1,3), cv2.COLOR_BGR2LAB)

    brightness = np.mean(hsv[:,0,2])
    redness = np.mean(lab[:,0,1])
    texture = np.var(region)

    return brightness, redness, texture


# -----------------------------
# Landmark indices (approximate)
# -----------------------------
LEFT_CHEEK = [234, 93, 132, 58, 172]
RIGHT_CHEEK = [454, 323, 361, 288, 397]
FOREHEAD = [10, 338, 297, 332, 284, 251]

# -----------------------------
# Load image (or webcam frame)
# -----------------------------
cap = cv2.VideoCapture(0)


time.sleep(2)

# Show live preview and wait for spacebar to capture
print("Press SPACEBAR to capture image, ESC to exit")
while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow('Press SPACEBAR to capture', image)
    
    key = cv2.waitKey(1)
    if key == 32:  # Spacebar
        print("Image captured!")
        break
    elif key == 27:  # ESC
        print("Cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()


# ret, image = cap.read()
# image = cv2.imread("face. jpg")  # replace with webcam frame if needed

# Convert to RGB (MediaPipe requires RGB)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create MediaPipe Image object
mp_image = mp. Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# Detect face landmarks
detection_result = detector.detect(mp_image)

if not detection_result.face_landmarks:
    print("No face detected")
    cap.release()
    exit()

# Get landmarks (new format)
landmarks = detection_result.face_landmarks[0]

regions = {}
regions["left_cheek"], _ = extract_region(image, landmarks, LEFT_CHEEK)
regions["right_cheek"], _ = extract_region(image, landmarks, RIGHT_CHEEK)
regions["forehead"], _ = extract_region(image, landmarks, FOREHEAD)

# -----------------------------
# Analyze regions
# -----------------------------
metrics = []
for name, region in regions.items():
    result = analyze_skin(region)
    if result:
        metrics. append(result)

# Average metrics
brightness, redness, texture = np.mean(metrics, axis=0)

# -----------------------------
# Simple rule-based classification
# -----------------------------
if brightness > 160 and texture < 500:
    oiliness = "Oily"
elif brightness < 120:
    oiliness = "Dry"
else:
    oiliness = "Normal"

redness_level = "High" if redness > 150 else "Low"

# -----------------------------
# Output results
# -----------------------------
print("Skin Assessment Results:")
print(f"• Oiliness: {oiliness}")
print(f"• Redness: {redness_level}")
print(f"• Texture variance: {texture:.2f}")

# Cleanup
cap.release()