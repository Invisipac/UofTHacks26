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
detector = vision.FaceLandmarker.create_from_options(options)

# -----------------------------
# Landmark indices for skin regions
# -----------------------------
# Expanded regions for better coverage
LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5]
CHIN = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148]

# Simplified forehead (top of face)
FOREHEAD = [10, 67, 109, 108, 69, 104, 68, 71, 21, 54, 103, 67, 109, 151, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401, 435, 288, 397]

# -----------------------------
# Color scheme for UI
# -----------------------------
COLORS = {
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'red': (0, 0, 255),
    'orange': (0, 165, 255),
    'white': (255, 255, 255),
    'blue': (255, 200, 0),
    'purple': (255, 0, 255),
    'panel_bg': (40, 40, 40),
}

# -----------------------------
# Blemish Tracking System
# -----------------------------
class BlemishTracker:
    """Tracks blemishes across frames for temporal consistency"""
    
    def __init__(self, confirmation_frames=8, max_distance=20, decay_frames=5):
        self.confirmation_frames = confirmation_frames  # Frames needed to confirm
        self.max_distance = max_distance  # Max pixels to consider same blemish
        self.decay_frames = decay_frames  # Frames before removing unconfirmed
        self.candidates = []  # [(x, y, r, frame_count, confirmed)]
        self.confirmed_blemishes = []  # Stable, confirmed blemishes
        
    def update(self, detected_blemishes):
        """Update tracker with newly detected blemishes"""
        # Match detected blemishes to existing candidates
        matched_indices = set()
        
        for (x, y, r) in detected_blemishes:
            matched = False
            for i, (cx, cy, cr, count, confirmed) in enumerate(self.candidates):
                # Check if this detection matches an existing candidate
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < self.max_distance:
                    # Update candidate position (moving average)
                    new_x = int(0.7 * cx + 0.3 * x)
                    new_y = int(0.7 * cy + 0.3 * y)
                    new_r = int(0.7 * cr + 0.3 * r)
                    new_count = min(count + 1, self.confirmation_frames + 5)
                    new_confirmed = confirmed or (new_count >= self.confirmation_frames)
                    self.candidates[i] = (new_x, new_y, new_r, new_count, new_confirmed)
                    matched_indices.add(i)
                    matched = True
                    break
            
            if not matched:
                # New candidate blemish
                self.candidates.append((x, y, r, 1, False))
        
        # Decay unmatched candidates
        updated_candidates = []
        for i, (cx, cy, cr, count, confirmed) in enumerate(self.candidates):
            if i in matched_indices:
                updated_candidates.append((cx, cy, cr, count, confirmed))
            else:
                # Decay the count
                new_count = count - 1
                if new_count > 0:
                    # Keep confirmed blemishes longer
                    if confirmed:
                        updated_candidates.append((cx, cy, cr, new_count, confirmed))
                    elif new_count > -self.decay_frames:
                        updated_candidates.append((cx, cy, cr, new_count, confirmed))
        
        self.candidates = updated_candidates
        
        # Get confirmed blemishes
        self.confirmed_blemishes = [
            (x, y, r) for (x, y, r, count, confirmed) in self.candidates
            if confirmed
        ]
        
        return self.confirmed_blemishes
    
    def get_confidence_blemishes(self):
        """Get blemishes with their confidence levels"""
        result = []
        for (x, y, r, count, confirmed) in self.candidates:
            confidence = min(1.0, count / self.confirmation_frames)
            result.append((x, y, r, confidence, confirmed))
        return result
    
    def reset(self):
        """Reset tracker"""
        self.candidates = []
        self.confirmed_blemishes = []


# Global blemish tracker
blemish_tracker = BlemishTracker()

# -----------------------------
# Pre-allocated kernels and CLAHE (optimization)
# -----------------------------
KERNEL_5x5 = np.ones((5, 5), dtype=np.uint8)
KERNEL_3x3 = np.ones((3, 3), dtype=np.uint8)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
LOWER_SKIN = np.array([0, 25, 50], dtype=np.uint8)
UPPER_SKIN = np.array([50, 255, 255], dtype=np.uint8)

# Frame skip counter for heavy operations
frame_counter = 0
SKIP_FRAMES = 2  # Process every Nth frame for blemish detection
cached_blemishes = []
cached_metrics = None

# -----------------------------
# Helper functions
# -----------------------------
def extract_region_fast(image, landmarks, indices, h, w):
    """Extract polygon region from facial landmarks (optimized)"""
    # Pre-filter valid indices
    valid_indices = [i for i in indices if i < len(landmarks)]
    if len(valid_indices) < 3:
        return None, None
    
    # Vectorized point extraction
    points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in valid_indices
    ], dtype=np.int32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    return mask, cv2.boundingRect(points)


def detect_blemishes_fast(image, mask, hsv_image, gray_image):
    """Detect blemishes with pre-computed color spaces (optimized)"""
    if mask is None:
        return [], 0
    
    # Get bounding box to process only ROI
    x, y, bw, bh = cv2.boundingRect(mask)
    if bw == 0 or bh == 0:
        return [], 0
    
    # Extract ROI for faster processing
    roi_mask = mask[y:y+bh, x:x+bw]
    roi_hsv = hsv_image[y:y+bh, x:x+bw]
    roi_gray = gray_image[y:y+bh, x:x+bw]
    
    # Skin color mask using pre-allocated bounds
    skin_color_mask = cv2.inRange(roi_hsv, LOWER_SKIN, UPPER_SKIN)
    combined_mask = cv2.bitwise_and(roi_mask, skin_color_mask)
    
    # Morphological ops - reduced erosion for more sensitivity
    combined_mask = cv2.erode(combined_mask, KERNEL_5x5, iterations=1)
    combined_mask = cv2.dilate(combined_mask, KERNEL_5x5, iterations=1)
    
    # Apply mask to gray ROI
    masked_gray = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)
    
    # Use pre-allocated CLAHE
    enhanced = CLAHE.apply(masked_gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)  # Smaller blur for more detail
    
    # Adaptive threshold - more sensitive settings
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 3)
    thresh = cv2.bitwise_and(thresh, thresh, mask=combined_mask)
    
    # Morphological cleanup with pre-allocated kernel
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, KERNEL_3x3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL_3x3)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blemishes = []
    pi_4 = 4 * np.pi  # Pre-compute
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # More sensitive: lower minimum area (25 instead of 40)
        if 25 < area < 450:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            # More sensitive: lower minimum radius (2.5 instead of 3)
            if 2.5 < radius < 14:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = pi_4 * area / (perimeter * perimeter)
                    # Slightly lower circularity threshold (0.35 instead of 0.4)
                    if circularity > 0.35:
                        px, py_local = int(cx), int(cy)
                        if 0 <= py_local < roi_hsv.shape[0] and 0 <= px < roi_hsv.shape[1]:
                            h_val, s_val, v_val = roi_hsv[py_local, px]
                            if 0 <= h_val <= 50 and s_val > 15 and v_val > 35:
                                # Offset back to full image coordinates
                                blemishes.append((int(cx) + x, int(cy) + y, int(radius)))
    
    return blemishes, min(100, len(blemishes) * 15)


def analyze_skin_fast(image, mask, hsv_image, lab_image):
    """Optimized skin analysis using pre-computed color spaces"""
    if mask is None:
        return None
    
    # Extract pixels using mask (vectorized)
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return None
    
    # Use pre-computed color spaces
    hsv_pixels = hsv_image[mask_bool]
    lab_pixels = lab_image[mask_bool]
    bgr_pixels = image[mask_bool]
    
    if len(hsv_pixels) == 0:
        return None
    
    # Extract channels directly (no reshape needed)
    s = hsv_pixels[:, 1].astype(np.float32)
    v = hsv_pixels[:, 2].astype(np.float32)
    l = lab_pixels[:, 0].astype(np.float32)
    a = lab_pixels[:, 1].astype(np.float32)
    
    # Compute metrics with numpy (vectorized)
    brightness = np.mean(v)
    shine_score = (brightness / 255) * 100
    s_mean = np.mean(s)
    
    oiliness_score = np.clip((shine_score - 40) * 1.5 + (s_mean / 255) * 30, 0, 100)
    
    texture_variance = np.var(bgr_pixels)
    dryness_score = np.clip((100 - shine_score) * 0.5 + (texture_variance / 1000) * 20, 0, 100)
    
    redness_score = np.clip((np.mean(a) - 128) * 3, 0, 100)
    
    uneven_tone_score = np.clip(np.var(l) / 5, 0, 100)
    
    dark_spot_score = np.clip((128 - np.percentile(l, 10)) * 1.5, 0, 100)
    
    return {
        'oiliness': float(oiliness_score),
        'dryness': float(dryness_score),
        'redness': float(redness_score),
        'uneven_tone': float(uneven_tone_score),
        'dark_spots': float(dark_spot_score),
    }


def calculate_product_ratios(skin_metrics, blemish_score):
    """
    Calculate dispenser ratios for 3 products based on skin analysis.
    
    Products:
    1. MOISTURIZER - For dryness, dehydration
    2. CLEANSER/OIL CONTROL - For oiliness, excess sebum
    3. TREATMENT SERUM - For acne, blemishes, redness, dark spots
    
    Returns ratios that sum to 100%
    """
    # Base scores for each product need
    moisturizer_need = skin_metrics['dryness'] * 1.2
    cleanser_need = skin_metrics['oiliness'] * 1.0
    treatment_need = (
        skin_metrics['redness'] * 0.4 +
        blemish_score * 0.4 +
        skin_metrics['dark_spots'] * 0.2
    )
    
    # Ensure minimum values
    moisturizer_need = max(10, moisturizer_need)
    cleanser_need = max(10, cleanser_need)
    treatment_need = max(10, treatment_need)
    
    # Normalize to 100%
    total = moisturizer_need + cleanser_need + treatment_need
    
    ratios = {
        'moisturizer': round((moisturizer_need / total) * 100),
        'cleanser': round((cleanser_need / total) * 100),
        'treatment': round((treatment_need / total) * 100)
    }
    
    # Ensure they sum to exactly 100
    diff = 100 - sum(ratios.values())
    ratios['moisturizer'] += diff
    
    return ratios


def get_severity_color(score):
    """Return color based on severity score (0-100)"""
    if score < 25:
        return COLORS['green']
    elif score < 50:
        return COLORS['yellow']
    elif score < 75:
        return COLORS['orange']
    else:
        return COLORS['red']


def get_severity_label(score):
    """Return severity label based on score"""
    if score < 20:
        return "Low"
    elif score < 40:
        return "Mild"
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "High"
    else:
        return "Severe"


def draw_overlay(image, landmarks, skin_data, blemishes_all, product_ratios):
    """Draw comprehensive overlay on the webcam feed"""
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Draw face mesh outline (subtle)
    face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                    132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
              for i in face_outline if i < len(landmarks)]
    if len(points) > 2:
        cv2.polylines(overlay, [np.array(points)], True, COLORS['blue'], 1)
    
    # Draw blemishes with confidence-based visualization
    confidence_blemishes = blemish_tracker.get_confidence_blemishes()
    for (x, y, r, confidence, confirmed) in confidence_blemishes:
        if confirmed:
            # Confirmed blemishes: solid red circle
            cv2.circle(overlay, (x, y), r + 2, COLORS['red'], 2)
            cv2.circle(overlay, (x, y), 2, COLORS['red'], -1)
        elif confidence > 0.4:
            # Candidate blemishes: yellow dashed (show as partial circle)
            cv2.circle(overlay, (x, y), r + 2, COLORS['yellow'], 1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Draw info panel on the right side
    panel_x = w - 280
    panel_y = 10
    panel_w = 270
    panel_h = 380
    
    # Semi-transparent panel background
    panel_overlay = image.copy()
    cv2.rectangle(panel_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  COLORS['panel_bg'], -1)
    cv2.addWeighted(panel_overlay, 0.85, image, 0.15, 0, image)
    
    # Panel border
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  COLORS['white'], 1)
    
    # Title
    cv2.putText(image, "SKIN ANALYSIS", (panel_x + 60, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 2)
    
    # Draw metrics with bars
    metrics_to_show = [
        ('Oiliness', skin_data['oiliness']),
        ('Dryness', skin_data['dryness']),
        ('Redness', skin_data['redness']),
        ('Blemishes', skin_data['blemish_score']),
        ('Uneven Tone', skin_data['uneven_tone']),
        ('Dark Spots', skin_data['dark_spots']),
    ]
    
    y_offset = panel_y + 55
    for label, score in metrics_to_show:
        # Label
        cv2.putText(image, f"{label}:", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['white'], 1)
        
        # Progress bar background
        bar_x = panel_x + 100
        bar_w = 120
        bar_h = 12
        cv2.rectangle(image, (bar_x, y_offset - 10), (bar_x + bar_w, y_offset + 2),
                      (60, 60, 60), -1)
        
        # Progress bar fill
        fill_w = int((score / 100) * bar_w)
        color = get_severity_color(score)
        cv2.rectangle(image, (bar_x, y_offset - 10), (bar_x + fill_w, y_offset + 2),
                      color, -1)
        
        # Score text
        cv2.putText(image, f"{int(score)}%", (bar_x + bar_w + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        y_offset += 28
    
    # Separator line
    y_offset += 5
    cv2.line(image, (panel_x + 10, y_offset), (panel_x + panel_w - 10, y_offset),
             COLORS['white'], 1)
    y_offset += 20
    
    # Product Ratios Section
    cv2.putText(image, "DISPENSER RATIOS", (panel_x + 50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['blue'], 2)
    y_offset += 30
    
    product_colors = {
        'moisturizer': (255, 200, 100),  # Light blue
        'cleanser': (100, 255, 100),      # Green
        'treatment': (100, 100, 255)       # Red/pink
    }
    
    product_labels = {
        'moisturizer': 'Moisturizer',
        'cleanser': 'Oil Control',
        'treatment': 'Treatment'
    }
    
    for product, ratio in product_ratios.items():
        color = product_colors[product]
        label = product_labels[product]
        
        cv2.putText(image, f"{label}:", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Ratio bar
        bar_x = panel_x + 100
        bar_w = 100
        cv2.rectangle(image, (bar_x, y_offset - 10), (bar_x + bar_w, y_offset + 2),
                      (60, 60, 60), -1)
        fill_w = int((ratio / 100) * bar_w)
        cv2.rectangle(image, (bar_x, y_offset - 10), (bar_x + fill_w, y_offset + 2),
                      color, -1)
        
        cv2.putText(image, f"{ratio}%", (bar_x + bar_w + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        y_offset += 25
    
    # Instructions at bottom
    cv2.putText(image, "SPACE: Capture | ESC: Exit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
    
    # Blemish count indicator - only show confirmed
    confirmed_blemishes = blemish_tracker.confirmed_blemishes
    blemish_count = len(confirmed_blemishes)
    candidate_count = len([b for b in blemish_tracker.get_confidence_blemishes() if not b[4] and b[3] > 0.4])
    
    if blemish_count > 0:
        cv2.putText(image, f"Confirmed spots: {blemish_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['red'], 2)
    if candidate_count > 0:
        cv2.putText(image, f"Detecting: {candidate_count}...", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['yellow'], 1)
    
    return image


# -----------------------------
# Main Application
# -----------------------------
def main():
    global frame_counter, cached_blemishes, cached_metrics
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    time.sleep(0.5)  # Shorter camera warmup
    
    print("=" * 50)
    print("SKIN ANALYSIS SYSTEM (Optimized)")
    print("=" * 50)
    print("Press SPACEBAR to capture final analysis")
    print("Press ESC to exit")
    print("=" * 50)
    
    # Initialize smoothing variables for stable display
    smooth_metrics = None
    smooth_alpha = 0.25  # Slightly less smoothing for responsiveness
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_counter += 1
        
        # Mirror the frame for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert color spaces once per frame (shared across all operations)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab_full = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect face landmarks
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # Extract region masks only (optimized)
            region_masks = {
                'left_cheek': extract_region_fast(frame, landmarks, LEFT_CHEEK, h, w),
                'right_cheek': extract_region_fast(frame, landmarks, RIGHT_CHEEK, h, w),
                'forehead': extract_region_fast(frame, landmarks, FOREHEAD, h, w),
                'nose': extract_region_fast(frame, landmarks, NOSE, h, w),
            }
            
            # Only do heavy blemish processing every N frames
            do_full_analysis = (frame_counter % SKIP_FRAMES == 0)
            
            all_metrics = []
            all_blemishes = []
            
            for name, result in region_masks.items():
                if result[0] is not None:
                    mask, bbox = result
                    
                    # Skin analysis (every frame for smooth metrics)
                    metrics = analyze_skin_fast(frame, mask, hsv_full, lab_full)
                    if metrics:
                        all_metrics.append(metrics)
                    
                    # Blemish detection (skip frames for performance)
                    if do_full_analysis:
                        blemishes, _ = detect_blemishes_fast(frame, mask, hsv_full, gray_full)
                        all_blemishes.extend(blemishes)
            
            # Use cached blemishes on skipped frames
            if do_full_analysis:
                cached_blemishes = all_blemishes
            else:
                all_blemishes = cached_blemishes
            
            # Update blemish tracker
            confirmed_blemishes = blemish_tracker.update(all_blemishes)
            
            if all_metrics:
                # Average metrics across all regions (optimized)
                n = len(all_metrics)
                avg_metrics = {
                    'oiliness': sum(m['oiliness'] for m in all_metrics) / n,
                    'dryness': sum(m['dryness'] for m in all_metrics) / n,
                    'redness': sum(m['redness'] for m in all_metrics) / n,
                    'uneven_tone': sum(m['uneven_tone'] for m in all_metrics) / n,
                    'dark_spots': sum(m['dark_spots'] for m in all_metrics) / n,
                    'blemish_score': min(100, len(confirmed_blemishes) * 12),
                }
                
                # Smooth metrics for stable display
                if smooth_metrics is None:
                    smooth_metrics = avg_metrics.copy()
                else:
                    inv_alpha = 1 - smooth_alpha
                    for key in avg_metrics:
                        smooth_metrics[key] = smooth_alpha * avg_metrics[key] + inv_alpha * smooth_metrics[key]
                
                # Calculate product ratios
                product_ratios = calculate_product_ratios(smooth_metrics, smooth_metrics['blemish_score'])
                
                # Draw overlay - pass confirmed blemishes only
                frame = draw_overlay(frame, landmarks, smooth_metrics, confirmed_blemishes, product_ratios)
        else:
            # No face detected
            cv2.putText(frame, "No face detected - Please face the camera", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['red'], 2)
        
        cv2.imshow('Skin Analysis System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar - capture
            if smooth_metrics:
                print("\n" + "=" * 50)
                print("FINAL SKIN ANALYSIS RESULTS")
                print("=" * 50)
                print(f"\n📊 SKIN METRICS:")
                print(f"   • Oiliness:    {smooth_metrics['oiliness']:.1f}% ({get_severity_label(smooth_metrics['oiliness'])})")
                print(f"   • Dryness:     {smooth_metrics['dryness']:.1f}% ({get_severity_label(smooth_metrics['dryness'])})")
                print(f"   • Redness:     {smooth_metrics['redness']:.1f}% ({get_severity_label(smooth_metrics['redness'])})")
                print(f"   • Blemishes:   {smooth_metrics['blemish_score']:.1f}% ({get_severity_label(smooth_metrics['blemish_score'])})")
                print(f"   • Uneven Tone: {smooth_metrics['uneven_tone']:.1f}% ({get_severity_label(smooth_metrics['uneven_tone'])})")
                print(f"   • Dark Spots:  {smooth_metrics['dark_spots']:.1f}% ({get_severity_label(smooth_metrics['dark_spots'])})")
                
                print(f"\n💧 RECOMMENDED DISPENSER RATIOS:")
                print(f"   • Moisturizer (Hydration):  {product_ratios['moisturizer']}%")
                print(f"   • Oil Control (Cleanser):   {product_ratios['cleanser']}%")
                print(f"   • Treatment (Acne/Repair):  {product_ratios['treatment']}%")
                
                # Product recommendations
                print(f"\n💡 RECOMMENDATIONS:")
                if smooth_metrics['oiliness'] > 50:
                    print("   → Use oil-free, mattifying products")
                if smooth_metrics['dryness'] > 50:
                    print("   → Increase hydration with hyaluronic acid")
                if smooth_metrics['redness'] > 40:
                    print("   → Look for calming ingredients (aloe, centella)")
                if smooth_metrics['blemish_score'] > 40:
                    print("   → Consider salicylic acid or benzoyl peroxide")
                if smooth_metrics['dark_spots'] > 40:
                    print("   → Use vitamin C or niacinamide serums")
                
                print("=" * 50)
                
                # Save the captured frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"skin_analysis_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n📸 Image saved as: {filename}")
            break
            
        elif key == 27:  # ESC - exit
            print("Cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()