"""
ACNE DETECTION SYSTEM v4 - Simplified Clear Spot Detection
Focus: Detect obvious acne spots (red/inflamed lesions, dark spots)
"""

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time

# Download model if needed
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded successfully!")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Face silhouette for mask
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Regions to analyze
FOREHEAD = [10, 67, 109, 108, 69, 104, 68, 71, 21, 54, 103, 151, 337, 299, 333, 298, 301, 338, 297]
LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 377, 400, 378, 365, 397]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 378, 400, 377, 148, 176, 149, 150, 136]
CHIN = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 175, 396, 369]
NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4]

# Exclusion zones (eyes, mouth)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Colors
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)


def get_polygon(landmarks, indices, h, w):
    """Convert landmark indices to polygon points"""
    valid = [i for i in indices if i < len(landmarks)]
    if len(valid) < 3:
        return None
    return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in valid], np.int32)


def create_face_mask(landmarks, h, w):
    """Create face mask excluding eyes and mouth"""
    face_pts = get_polygon(landmarks, FACE_OUTLINE, h, w)
    if face_pts is None:
        return None
    
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [face_pts], 255)
    
    # Exclude eyes
    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        eye_pts = get_polygon(landmarks, eye_idx, h, w)
        if eye_pts is not None:
            hull = cv2.convexHull(eye_pts)
            # Dilate eye region
            eye_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(eye_mask, hull, 255)
            eye_mask = cv2.dilate(eye_mask, np.ones((15, 15), np.uint8))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(eye_mask))
    
    # Exclude mouth
    mouth_pts = get_polygon(landmarks, MOUTH, h, w)
    if mouth_pts is not None:
        hull = cv2.convexHull(mouth_pts)
        mouth_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mouth_mask, hull, 255)
        mouth_mask = cv2.dilate(mouth_mask, np.ones((10, 10), np.uint8))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(mouth_mask))
    
    return mask


def detect_acne_simple(image, face_mask):
    """
    Simple acne detection - look for clear red spots and dark spots.
    No fancy preprocessing, just direct color analysis.
    """
    if face_mask is None:
        return []
    
    h, w = image.shape[:2]
    spots = []
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)  # Lightness
    a_ch = lab[:, :, 1].astype(np.float32)  # Red-Green (higher = more red)
    
    # Get face region stats
    face_pixels_a = a_ch[face_mask > 0]
    face_pixels_l = l_ch[face_mask > 0]
    
    if len(face_pixels_a) == 0:
        return []
    
    a_mean = np.mean(face_pixels_a)
    a_std = np.std(face_pixels_a)
    l_mean = np.mean(face_pixels_l)
    l_std = np.std(face_pixels_l)
    
    # === DETECT RED/INFLAMED SPOTS ===
    # Simple: find pixels significantly more red than average skin
    red_threshold = a_mean + (a_std * 1.5)  # 1.5 std above mean = clearly red
    red_mask = ((a_ch > red_threshold) & (face_mask > 0)).astype(np.uint8) * 255
    
    # === DETECT DARK SPOTS ===
    # Simple: find pixels significantly darker than average skin
    dark_threshold = l_mean - (l_std * 1.8)  # 1.8 std below mean = clearly dark
    dark_mask = ((l_ch < dark_threshold) & (face_mask > 0)).astype(np.uint8) * 255
    
    # Combine
    combined = cv2.bitwise_or(red_mask, dark_mask)
    
    # Light cleanup - just remove tiny noise
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Only consider spots of reasonable size (not tiny noise, not huge areas)
        if 25 < area < 500:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            if 3 < radius < 15:  # Reasonable spot size
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Acne is generally round-ish
                    if circularity > 0.4:
                        px, py = int(cx), int(cy)
                        
                        # Check redness at this spot
                        if 0 <= py < h and 0 <= px < w:
                            redness = (a_ch[py, px] - a_mean) / max(a_std, 1)
                            darkness = (l_mean - l_ch[py, px]) / max(l_std, 1)
                            intensity = max(redness, darkness) / 3  # Normalize to ~0-1
                            intensity = min(1.0, max(0.3, intensity))
                            
                            spots.append((px, py, int(radius), intensity))
    
    # Remove duplicates
    filtered = []
    for spot in spots:
        is_dup = False
        for existing in filtered:
            dist = np.sqrt((spot[0] - existing[0])**2 + (spot[1] - existing[1])**2)
            if dist < 15:
                is_dup = True
                break
        if not is_dup:
            filtered.append(spot)
    
    return filtered


def classify_spots(spots, landmarks, h, w):
    """Classify spots by facial region"""
    regions = {
        'forehead': get_polygon(landmarks, FOREHEAD, h, w),
        'left_cheek': get_polygon(landmarks, LEFT_CHEEK, h, w),
        'right_cheek': get_polygon(landmarks, RIGHT_CHEEK, h, w),
        'chin': get_polygon(landmarks, CHIN, h, w),
        'nose': get_polygon(landmarks, NOSE, h, w)
    }
    
    counts = {name: 0 for name in regions}
    
    for (x, y, r, intensity) in spots:
        for name, pts in regions.items():
            if pts is not None:
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    counts[name] += 1
                    break
    
    return counts


def calculate_product_ratios(count, avg_intensity):
    """Calculate dispenser ratios"""
    if count == 0:
        return {'cleanser': 35, 'treatment': 15, 'moisturizer': 50}
    elif count < 3:
        return {'cleanser': 35, 'treatment': 30, 'moisturizer': 35}
    elif count < 6:
        return {'cleanser': 30, 'treatment': 45, 'moisturizer': 25}
    elif count < 10:
        return {'cleanser': 25, 'treatment': 55, 'moisturizer': 20}
    else:
        return {'cleanser': 20, 'treatment': 65, 'moisturizer': 15}


def draw_overlay(frame, landmarks, spots, region_counts, ratios):
    """Draw results on image"""
    h, w = frame.shape[:2]
    output = frame.copy()
    
    # Face outline
    face_pts = get_polygon(landmarks, FACE_OUTLINE, h, w)
    if face_pts is not None:
        cv2.polylines(output, [face_pts], True, CYAN, 1)
    
    # Draw spots
    for (x, y, r, intensity) in spots:
        # Red = high intensity, Yellow = low intensity
        color = (0, int(255 * (1 - intensity)), int(255 * intensity))
        cv2.circle(output, (x, y), r + 3, color, 2)
        cv2.circle(output, (x, y), 2, RED, -1)
    
    # Info panel
    panel_x, panel_y = w - 250, 10
    panel_w, panel_h = 240, 300
    
    overlay = output.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
    cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), WHITE, 1)
    
    # Title
    cv2.putText(output, "ACNE DETECTION", (panel_x + 50, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
    
    total = len(spots)
    
    # Severity
    if total == 0:
        sev, col = "Clear", GREEN
    elif total < 5:
        sev, col = "Mild", GREEN
    elif total < 10:
        sev, col = "Moderate", YELLOW
    elif total < 15:
        sev, col = "Notable", ORANGE
    else:
        sev, col = "Severe", RED
    
    cv2.putText(output, f"Spots: {total}  |  {sev}", (panel_x + 15, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    
    # Region counts
    cv2.line(output, (panel_x + 10, panel_y + 70), (panel_x + panel_w - 10, panel_y + 70), WHITE, 1)
    cv2.putText(output, "BY REGION:", (panel_x + 15, panel_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    
    y_off = panel_y + 115
    for region, count in region_counts.items():
        label = region.replace('_', ' ').title()
        cv2.putText(output, f"{label}: {count}", (panel_x + 20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        y_off += 20
    
    # Product ratios
    cv2.line(output, (panel_x + 10, y_off + 5), (panel_x + panel_w - 10, y_off + 5), WHITE, 1)
    cv2.putText(output, "DISPENSER:", (panel_x + 15, y_off + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
    
    y_off += 50
    for product, ratio in ratios.items():
        color = (100, 255, 100) if product == 'cleanser' else (100, 100, 255) if product == 'treatment' else (255, 200, 100)
        cv2.putText(output, f"{product.title()}: {ratio}%", (panel_x + 20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # Bar
        bar_w = int(ratio * 0.8)
        cv2.rectangle(output, (panel_x + 120, y_off - 10), (panel_x + 120 + bar_w, y_off), color, -1)
        y_off += 22
    
    return output


def analyze_image(image_path):
    """Analyze image for acne"""
    print("=" * 50)
    print("   ACNE DETECTION v4 - Clear Spot Detection")
    print("=" * 50)
    print(f"  File: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("  ERROR: Could not load image")
        return None
    
    h, w = frame.shape[:2]
    
    # Resize if needed
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w = frame.shape[:2]
    
    # Detect face
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    
    if not result.face_landmarks:
        print("  ERROR: No face detected")
        return None
    
    landmarks = result.face_landmarks[0]
    face_mask = create_face_mask(landmarks, h, w)
    
    # Detect acne
    spots = detect_acne_simple(frame, face_mask)
    region_counts = classify_spots(spots, landmarks, h, w)
    
    avg_intensity = sum(s[3] for s in spots) / len(spots) if spots else 0.5
    ratios = calculate_product_ratios(len(spots), avg_intensity)
    
    # Draw overlay
    output = draw_overlay(frame, landmarks, spots, region_counts, ratios)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"  SPOTS DETECTED: {len(spots)}")
    print("-" * 50)
    for region, count in region_counts.items():
        print(f"    {region.replace('_', ' ').title():15} {count}")
    print("-" * 50)
    print(f"  DISPENSER: Cleanser {ratios['cleanser']}% | Treatment {ratios['treatment']}% | Moisturizer {ratios['moisturizer']}%")
    print("=" * 50)
    
    # Save
    base = os.path.splitext(os.path.basename(image_path))[0]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"{base}_result_{ts}.jpg"
    cv2.imwrite(out_file, output)
    print(f"  Saved: {out_file}")
    
    # Show
    cv2.imshow('Acne Detection Result', output)
    print("  Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {'spots': len(spots), 'regions': region_counts, 'ratios': ratios}


def main():
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("=" * 50)
        print("   ACNE DETECTION v4")
        print("=" * 50)
        print("\n  Usage: python extract_details.py <image_path>")
        image_path = input("\n  Image path: ").strip().strip('"').strip("'")
    
    if not image_path or not os.path.exists(image_path):
        print("  ERROR: Invalid file path")
        return
    
    analyze_image(image_path)


if __name__ == "__main__":
    main()