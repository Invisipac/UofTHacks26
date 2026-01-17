import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

# -----------------------------
# Download model if needed
# -----------------------------
model_path = 'face_landmarker. task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
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
detector = vision.FaceLandmarker. create_from_options(options)

# -----------------------------
# Helper functions
# -----------------------------
def extract_region(image, landmarks, indices):
    """Extract polygon region from facial landmarks"""
    h, w, _ = image.shape
    points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
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


def detect_blemishes(image, landmarks, red_sensitivity=70, dark_threshold=60, texture_sensitivity=15):
    """Detect and highlight blemishes/acne on face with adjustable parameters"""
    h, w, _ = image.shape
    
    # Define face region (excluding eyes, eyebrows, mouth)
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    # Create mask for face region
    points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in FACE_OVAL
    ])
    
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [points], 255)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Method 1: Detect STRONG redness (actual inflammation)
    # Use adjustable red_sensitivity parameter
    lower_red1 = np.array([0, red_sensitivity, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, red_sensitivity, 50])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Use Cr channel from YCrCb for better redness detection
    _, cr_mask = cv2.threshold(ycrcb[: ,:,1], 150, 255, cv2.THRESH_BINARY)
    
    # Combine red detection methods
    red_combined = cv2.bitwise_and(red_mask, cr_mask)
    
    # Method 2: Detect prominent dark spots
    # Use adjustable dark_threshold parameter
    _, dark_mask = cv2.threshold(gray_filtered, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Use adaptive threshold to avoid false positives from lighting
    adaptive_dark = cv2.adaptiveThreshold(
        gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 5
    )
    
    # Only keep very dark spots
    dark_mask = cv2.bitwise_and(dark_mask, adaptive_dark)
    
    # Method 3: Detect texture irregularities (bumps)
    # Use difference of Gaussians to detect raised areas
    blur1 = cv2.GaussianBlur(gray, (5, 5), 1)
    blur2 = cv2.GaussianBlur(gray, (9, 9), 2)
    dog = cv2.absdiff(blur1, blur2)
    
    # Use adjustable texture_sensitivity parameter
    _, texture_mask = cv2.threshold(dog, texture_sensitivity, 255, cv2.THRESH_BINARY)
    
    # Method 4: Use Laplacian for edge detection (pimple edges)
    laplacian = cv2.Laplacian(gray_filtered, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, edge_mask = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    # Combine methods with different weights
    # Red spots are most important indicator
    combined_mask = cv2.bitwise_or(red_combined, dark_mask)
    
    # Add texture only where there's already some redness or darkness
    texture_areas = cv2.bitwise_and(texture_mask, combined_mask)
    combined_mask = cv2.bitwise_or(combined_mask, texture_areas)
    
    # Only keep detections within face region
    combined_mask = cv2.bitwise_and(combined_mask, face_mask)
    
    # Clean up noise - more aggressive cleaning
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    
    # Remove tiny noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Fill small holes in detections
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Remove small isolated pixels
    combined_mask = cv2.medianBlur(combined_mask, 5)
    
    # Final erosion to reduce false positives
    combined_mask = cv2.erode(combined_mask, kernel_small, iterations=1)
    
    return combined_mask


def highlight_blemishes(image, blemish_mask, min_size=50, max_size=1500, circularity_threshold=30):
    """Draw circles around detected blemishes with adjustable parameters"""
    result = image.copy()
    
    # Find contours of blemishes
    contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blemish_count = 0
    blemish_sizes = []
    
    # Convert circularity from 0-100 scale to 0.0-1.0
    circ_thresh = circularity_threshold / 100.0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Use adjustable size filtering
        if min_size < area < max_size:
            # Get center and radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Additional check: must be somewhat circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Only highlight if reasonably circular (actual blemishes tend to be round)
                if circularity > circ_thresh: 
                    # Draw circle around blemish
                    cv2.circle(result, center, radius + 3, (0, 0, 255), 2)
                    blemish_count += 1
                    blemish_sizes.append(area)
    
    return result, blemish_count


def create_blemish_heatmap(image, blemish_mask):
    """Create a heatmap overlay showing blemish severity"""
    # Apply Gaussian blur to create smooth heatmap
    heatmap = cv2.GaussianBlur(blemish_mask, (21, 21), 0)
    
    # Convert to color heatmap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay on original image
    overlay = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
    
    return overlay


# -----------------------------
# Landmark indices (approximate)
# -----------------------------
LEFT_CHEEK = [234, 93, 132, 58, 172]
RIGHT_CHEEK = [454, 323, 361, 288, 397]
FOREHEAD = [10, 338, 297, 332, 284, 251]

# -----------------------------
# Real-time webcam processing
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("\n" + "="*60)
print("REAL-TIME BLEMISH DETECTION WITH ADJUSTABLE PARAMETERS")
print("="*60)
print("Controls:")
print("  SPACEBAR - Capture and analyze current frame")
print("  H - Toggle heatmap view")
print("  C - Toggle circles view")
print("  R - Reset sliders to default values")
print("  ESC - Exit")
print("\nAdjustable Sliders:")
print("  Red Sensitivity - Higher = only very red spots (50-150)")
print("  Dark Threshold - Lower = more sensitive to dark spots (30-100)")
print("  Texture Sensitivity - Higher = more texture detection (5-50)")
print("  Min Size - Minimum blemish size in pixels (10-200)")
print("  Max Size - Maximum blemish size in pixels (500-3000)")
print("  Circularity - How round must blemishes be (0-100, 100=perfect circle)")
print("="*60 + "\n")

# Create window and trackbars
window_name = 'Real-Time Blemish Detection'
cv2.namedWindow(window_name)

# Create trackbars for tuning (with default values)
cv2.createTrackbar('Red Sensitivity', window_name, 70, 150, lambda x: None)
cv2.createTrackbar('Dark Threshold', window_name, 60, 100, lambda x: None)
cv2.createTrackbar('Texture Sensitivity', window_name, 15, 50, lambda x: None)
cv2.createTrackbar('Min Size', window_name, 50, 200, lambda x: None)
cv2.createTrackbar('Max Size', window_name, 1500, 3000, lambda x: None)
cv2.createTrackbar('Circularity', window_name, 30, 100, lambda x: None)

# Toggle states
show_heatmap = False
show_circles = True

# Default values for reset
default_values = {
    'Red Sensitivity': 70,
    'Dark Threshold': 60,
    'Texture Sensitivity':  15,
    'Min Size': 50,
    'Max Size': 1500,
    'Circularity': 30
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Get current trackbar values
    red_sens = cv2.getTrackbarPos('Red Sensitivity', window_name)
    dark_thresh = cv2.getTrackbarPos('Dark Threshold', window_name)
    texture_sens = cv2.getTrackbarPos('Texture Sensitivity', window_name)
    min_size = cv2.getTrackbarPos('Min Size', window_name)
    max_size = cv2.getTrackbarPos('Max Size', window_name)
    circularity = cv2.getTrackbarPos('Circularity', window_name)
    
    # Ensure min values aren't zero
    if red_sens == 0:
        red_sens = 1
    if dark_thresh == 0:
        dark_thresh = 1
    if texture_sens == 0:
        texture_sens = 1
    if min_size == 0:
        min_size = 1
    if max_size == 0:
        max_size = 500
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat. SRGB, data=rgb)
    
    # Detect face landmarks
    detection_result = detector.detect(mp_image)
    
    display_frame = frame.copy()
    blemish_count = 0
    
    if detection_result. face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        
        # Detect blemishes with adjustable parameters
        blemish_mask = detect_blemishes(frame, landmarks, red_sens, dark_thresh, texture_sens)
        
        # Apply visualization based on toggle states
        if show_heatmap:
            display_frame = create_blemish_heatmap(frame, blemish_mask)
        
        if show_circles:
            if show_heatmap:
                # Apply circles on top of heatmap
                display_frame, blemish_count = highlight_blemishes(
                    display_frame, blemish_mask, min_size, max_size, circularity
                )
            else:
                # Apply circles on original frame
                display_frame, blemish_count = highlight_blemishes(
                    frame, blemish_mask, min_size, max_size, circularity
                )
        else:
            # Still count blemishes even if not showing circles
            contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circ_thresh = circularity / 100.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_size < area < max_size:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circ = 4 * np. pi * area / (perimeter * perimeter)
                        if circ > circ_thresh: 
                            blemish_count += 1
        
        # Determine severity
        if blemish_count == 0:
            severity = "Clear"
            severity_color = (0, 255, 0)  # Green
        elif blemish_count < 5:
            severity = "Mild"
            severity_color = (0, 255, 255)  # Yellow
        elif blemish_count < 15:
            severity = "Moderate"
            severity_color = (0, 165, 255)  # Orange
        else:
            severity = "Severe"
            severity_color = (0, 0, 255)  # Red
        
        # Display info on frame
        cv2.putText(display_frame, f"Blemishes: {blemish_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Severity: {severity}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, severity_color, 2)
        
        # Show current view mode
        view_mode = []
        if show_circles:
            view_mode.append("Circles")
        if show_heatmap:
            view_mode.append("Heatmap")
        if not view_mode:
            view_mode.append("Normal")
        
        cv2.putText(display_frame, f"View: {' + '.join(view_mode)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show current parameter values (smaller text)
        y_offset = 120
        cv2.putText(display_frame, f"Red:{red_sens} Dark:{dark_thresh} Tex:{texture_sens}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(display_frame, f"Size:{min_size}-{max_size} Circ:{circularity}", 
                    (10, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    else:
        cv2.putText(display_frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Show controls
    cv2.putText(display_frame, "SPACE:  Analyze | H:  Heatmap | C: Circles | R: Reset | ESC: Exit", 
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display the frame
    cv2.imshow(window_name, display_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\nExiting...")
        break
    
    elif key == ord('h') or key == ord('H'):  # Toggle heatmap
        show_heatmap = not show_heatmap
        print(f"Heatmap:  {'ON' if show_heatmap else 'OFF'}")
    
    elif key == ord('c') or key == ord('C'):  # Toggle circles
        show_circles = not show_circles
        print(f"Circles:  {'ON' if show_circles else 'OFF'}")
    
    elif key == ord('r') or key == ord('R'):  # Reset sliders
        print("\nResetting sliders to default values...")
        for param, value in default_values.items():
            cv2.setTrackbarPos(param, window_name, value)
        print("Sliders reset!")
    
    elif key == 32:  # Spacebar - Capture and full analysis
        print("\n" + "="*60)
        print("CAPTURING FRAME FOR DETAILED ANALYSIS...")
        print("="*60)
        
        # Save current frame
        cv2.imwrite("captured_frame.jpg", frame)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # Generate all visualizations with current parameters
            blemish_mask = detect_blemishes(frame, landmarks, red_sens, dark_thresh, texture_sens)
            highlighted_image, blemish_count = highlight_blemishes(
                frame, blemish_mask, min_size, max_size, circularity
            )
            heatmap_image = create_blemish_heatmap(frame, blemish_mask)
            
            # Save visualizations
            cv2.imwrite("blemishes_highlighted.jpg", highlighted_image)
            cv2.imwrite("blemishes_heatmap.jpg", heatmap_image)
            cv2.imwrite("blemishes_mask.jpg", blemish_mask)
            
            # Analyze skin regions
            regions = {}
            regions["left_cheek"], _ = extract_region(frame, landmarks, LEFT_CHEEK)
            regions["right_cheek"], _ = extract_region(frame, landmarks, RIGHT_CHEEK)
            regions["forehead"], _ = extract_region(frame, landmarks, FOREHEAD)
            
            metrics = []
            for name, region in regions.items():
                result = analyze_skin(region)
                if result:
                    metrics.append(result)
            
            if metrics:
                brightness, redness, texture = np.mean(metrics, axis=0)
                
                # Classifications
                if brightness > 160 and texture < 500:
                    oiliness = "Oily"
                elif brightness < 120: 
                    oiliness = "Dry"
                else:
                    oiliness = "Normal"
                
                redness_level = "High" if redness > 150 else "Low"
                
                if blemish_count == 0:
                    severity = "Clear"
                elif blemish_count < 5:
                    severity = "Mild"
                elif blemish_count < 15:
                    severity = "Moderate"
                else:
                    severity = "Severe"
                
                # Print results
                print("\nSKIN ASSESSMENT RESULTS")
                print("="*60)
                print(f"• Oiliness: {oiliness}")
                print(f"• Redness: {redness_level}")
                print(f"• Texture variance: {texture:.2f}")
                print(f"• Blemishes detected: {blemish_count}")
                print(f"• Acne severity: {severity}")
                print("="*60)
                
                print(f"\nRaw metrics:")
                print(f"• Brightness: {brightness:.2f}")
                print(f"• Redness value: {redness:.2f}")
                
                print(f"\nCurrent detection parameters:")
                print(f"• Red Sensitivity: {red_sens}")
                print(f"• Dark Threshold: {dark_thresh}")
                print(f"• Texture Sensitivity: {texture_sens}")
                print(f"• Min Size: {min_size}")
                print(f"• Max Size: {max_size}")
                print(f"• Circularity: {circularity}")
                
                print("\nSaved files:")
                print("  - captured_frame.jpg")
                print("  - blemishes_highlighted.jpg")
                print("  - blemishes_heatmap.jpg")
                print("  - blemishes_mask.jpg")
                print("="*60 + "\n")
        else:
            print("No face detected in captured frame")

cap.release()
cv2.destroyAllWindows()

print("\nThank you for using the skin analysis tool!")