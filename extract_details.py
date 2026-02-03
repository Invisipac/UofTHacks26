"""
SKIN ANALYSIS SYSTEM v5 - Multi-Feature Detection
Detects: Acne (red), Other Blemishes (blue), Oily Zones (yellow overlay)
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

# Exclusion zones (eyes, mouth, nostrils)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSTRILS = [94, 141, 242, 97, 2, 326, 462, 370, 94]  # Nostril area around nose tip

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


def detect_blemishes(image, face_mask, landmarks, h, w):
    """
    Detect other blemishes (non-acne): hyperpigmentation, texture irregularities, scars.
    Uses brown/tan color detection and texture variance analysis.
    Excludes eye and nostril areas with expanded margins.
    """
    if face_mask is None:
        return []
    
    blemishes = []
    
    # Create exclusion mask for eyes and nostrils
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Fill exclusion zones
    left_eye_pts = get_polygon(landmarks, LEFT_EYE, h, w)
    right_eye_pts = get_polygon(landmarks, RIGHT_EYE, h, w)
    nostril_pts = get_polygon(landmarks, NOSTRILS, h, w)
    
    if left_eye_pts is not None:
        cv2.fillPoly(exclusion_mask, [left_eye_pts], 255)
    if right_eye_pts is not None:
        cv2.fillPoly(exclusion_mask, [right_eye_pts], 255)
    if nostril_pts is not None:
        cv2.fillPoly(exclusion_mask, [nostril_pts], 255)
    
    # Dilate exclusion zones to expand the margins significantly
    expand_kernel = np.ones((25, 25), np.uint8)  # Large kernel for significant expansion
    exclusion_mask = cv2.dilate(exclusion_mask, expand_kernel, iterations=2)
    
    # Create blemish mask by subtracting exclusion zones from face mask
    blemish_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(exclusion_mask))
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)  # Lightness
    b_ch = lab[:, :, 2].astype(np.float32)  # Blue-Yellow (higher = more yellow/brown)
    
    # Get face region stats (using full face mask for baseline)
    face_pixels_l = l_ch[face_mask > 0]
    face_pixels_b = b_ch[face_mask > 0]
    
    if len(face_pixels_l) == 0:
        return []
    
    l_mean = np.mean(face_pixels_l)
    l_std = np.std(face_pixels_l)
    b_mean = np.mean(face_pixels_b)
    b_std = np.std(face_pixels_b)
    
    # === DETECT HYPERPIGMENTATION (brown/tan spots) ===
    # Higher b* = more yellow/brown, combined with darker areas
    brown_threshold = b_mean + (b_std * 1.2)
    slightly_dark = l_mean - (l_std * 0.8)
    brown_mask = ((b_ch > brown_threshold) & (l_ch < slightly_dark) & (blemish_mask > 0)).astype(np.uint8) * 255
    
    # === DETECT TEXTURE IRREGULARITIES ===
    # Use local variance to find rough/textured patches
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (15, 15))
    local_sq_mean = cv2.blur(gray**2, (15, 15))
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)
    
    face_var = local_var[face_mask > 0]
    if len(face_var) > 0:
        var_mean = np.mean(face_var)
        var_std = np.std(face_var)
        texture_threshold = var_mean + (var_std * 2.0)
        texture_mask = ((local_var > texture_threshold) & (blemish_mask > 0)).astype(np.uint8) * 255
    else:
        texture_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Combine blemish detections
    combined = cv2.bitwise_or(brown_mask, texture_mask)
    
    # Cleanup
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Blemishes can be larger than acne spots
        if 30 < area < 800:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            if 3 < radius < 20:
                px, py = int(cx), int(cy)
                
                if 0 <= py < h and 0 <= px < w:
                    # Calculate severity based on color deviation
                    brownness = (b_ch[py, px] - b_mean) / max(b_std, 1)
                    severity = min(1.0, max(0.3, brownness / 2))
                    
                    blemishes.append((px, py, int(radius), severity))
    
    # Remove duplicates
    filtered = []
    for b in blemishes:
        is_dup = False
        for existing in filtered:
            dist = np.sqrt((b[0] - existing[0])**2 + (b[1] - existing[1])**2)
            if dist < 20:
                is_dup = True
                break
        if not is_dup:
            filtered.append(b)
    
    return filtered


def detect_oiliness(image, face_mask, landmarks, h, w):
    """
    Detect skin oiliness by looking for specular highlights (shiny areas).
    Focuses on T-zone (forehead, nose, chin) where oiliness is most common.
    Returns oily regions and overall oiliness score.
    """
    if face_mask is None:
        return [], 0.0
    
    # Define T-zone mask
    tzone_mask = np.zeros((h, w), dtype=np.uint8)
    
    forehead_pts = get_polygon(landmarks, FOREHEAD, h, w)
    nose_pts = get_polygon(landmarks, NOSE, h, w)
    chin_pts = get_polygon(landmarks, CHIN, h, w)
    
    if forehead_pts is not None:
        cv2.fillPoly(tzone_mask, [forehead_pts], 255)
    if nose_pts is not None:
        cv2.fillPoly(tzone_mask, [nose_pts], 255)
    if chin_pts is not None:
        cv2.fillPoly(tzone_mask, [chin_pts], 255)
    
    # Combine with face mask
    tzone_mask = cv2.bitwise_and(tzone_mask, face_mask)
    
    # Convert to LAB for lightness analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    
    # Also look at saturation - oily skin reflects more (lower saturation in highlights)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)
    
    # Get T-zone stats
    tzone_pixels_l = l_ch[tzone_mask > 0]
    tzone_pixels_s = s_ch[tzone_mask > 0]
    tzone_pixels_v = v_ch[tzone_mask > 0]
    
    if len(tzone_pixels_l) == 0:
        return [], 0.0
    
    l_mean = np.mean(tzone_pixels_l)
    l_std = np.std(tzone_pixels_l)
    s_mean = np.mean(tzone_pixels_s)
    v_mean = np.mean(tzone_pixels_v)
    v_std = np.std(tzone_pixels_v)
    
    # Detect shine: high brightness + low saturation (specular highlights)
    bright_threshold = l_mean + (l_std * 1.5)
    low_sat_threshold = s_mean * 0.7  # 30% below average saturation
    
    # Shine mask: bright areas with low saturation
    shine_mask = ((l_ch > bright_threshold) & (s_ch < low_sat_threshold) & (tzone_mask > 0)).astype(np.uint8) * 255
    
    # Also detect based on high value channel
    high_v_mask = ((v_ch > v_mean + v_std * 1.8) & (tzone_mask > 0)).astype(np.uint8) * 255
    
    # Combine
    oily_mask = cv2.bitwise_or(shine_mask, high_v_mask)
    
    # Cleanup - smooth out the detection
    kernel = np.ones((5, 5), np.uint8)
    oily_mask = cv2.morphologyEx(oily_mask, cv2.MORPH_CLOSE, kernel)
    oily_mask = cv2.morphologyEx(oily_mask, cv2.MORPH_OPEN, kernel)
    
    # Find oily regions
    contours, _ = cv2.findContours(oily_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    oily_regions = []
    total_oily_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area for oily region
            oily_regions.append(contour)
            total_oily_area += area
    
    # Calculate oiliness score (0-100)
    tzone_area = np.sum(tzone_mask > 0)
    if tzone_area > 0:
        oiliness_score = min(100, (total_oily_area / tzone_area) * 200)  # Scale to 0-100
    else:
        oiliness_score = 0
    
    return oily_regions, oiliness_score


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


def calculate_product_ratios(acne_count, blemish_count, oiliness_score):
    """
    Calculate dispenser ratios using skin severity algorithm.
    
    Formula:
        Severity Score = (acne × 3) + (blemishes × 1.5) + (oiliness% × 0.3)
        
        Base ratios: toner 33%, Treatment 17%, Moisturizer 50%
        
        Adjustments:
        - Treatment: +1.5% per severity point (capped at 55%)
        - Moisturizer: -1% per severity point, -0.2% per oiliness% (min 15%)
        - toner: fills remainder (typically 25-35%)
        
        Total dispense: 3.5ml base, +0.5ml if oiliness > 50%
    
    Returns:
        dict with 'toner', 'treatment', 'moisturizer' percentages and 'total_ml'
    """
    # Calculate severity score
    severity = (acne_count * 3) + (blemish_count * 1.5) + (oiliness_score * 0.3)
    
    # Base ratios
    base_treatment = 17
    base_moisturizer = 50
    
    # Calculate treatment (increases with severity, capped at 55%)
    treatment = base_treatment + (severity * 1.5)
    treatment = min(55, max(17, treatment))  # Clamp between 17-55%
    
    # Calculate moisturizer (decreases with severity and oiliness, min 15%)
    moisturizer = base_moisturizer - (severity * 1.0) - (oiliness_score * 0.2)
    moisturizer = min(50, max(15, moisturizer))  # Clamp between 15-50%
    
    # toner fills the remainder
    toner = 100 - treatment - moisturizer
    toner = min(40, max(20, toner))  # Clamp between 20-40%
    
    # Normalize to ensure they sum to 100%
    total = toner + treatment + moisturizer
    toner = round(toner / total * 100)
    treatment = round(treatment / total * 100)
    moisturizer = 100 - toner - treatment  # Ensure exact 100%
    
    # Calculate total ml to dispense
    total_ml = 3.5
    if oiliness_score > 50:
        total_ml = 4.0  # Extra for oily skin
    
    return {
        'toner': toner,
        'treatment': treatment,
        'moisturizer': moisturizer,
        'total_ml': total_ml,
        'severity_score': round(severity, 1)
    }


def draw_overlay(frame, landmarks, spots, blemishes, oily_regions, oiliness_score, region_counts):
    """Draw results on image with all detections"""
    h, w = frame.shape[:2]
    output = frame.copy()
    
    # Face outline
    face_pts = get_polygon(landmarks, FACE_OUTLINE, h, w)
    if face_pts is not None:
        cv2.polylines(output, [face_pts], True, CYAN, 1)
    
    # Draw oily regions (yellow shaded overlay) - draw first so spots appear on top
    if oily_regions:
        oily_overlay = output.copy()
        for contour in oily_regions:
            cv2.fillPoly(oily_overlay, [contour], (0, 200, 255))  # Yellow/orange tint
        cv2.addWeighted(oily_overlay, 0.3, output, 0.7, 0, output)
        # Draw outlines
        for contour in oily_regions:
            cv2.drawContours(output, [contour], -1, YELLOW, 1)
    
    # Draw blemishes (blue circles)
    BLUE = (255, 150, 0)  # BGR - blue with slight cyan
    for (x, y, r, severity) in blemishes:
        color = (int(255 * severity), int(150 * severity), 0)  # Darker blue = more severe
        cv2.circle(output, (x, y), r + 3, BLUE, 2)
        cv2.circle(output, (x, y), 2, (255, 100, 0), -1)  # Blue center dot
    
    # Draw acne spots (red circles) - on top
    for (x, y, r, intensity) in spots:
        # Red = high intensity, Yellow = low intensity
        color = (0, int(255 * (1 - intensity)), int(255 * intensity))
        cv2.circle(output, (x, y), r + 3, color, 2)
        cv2.circle(output, (x, y), 2, RED, -1)
    
    # Info panel
    panel_x, panel_y = w - 280, 10
    panel_w, panel_h = 270, 320
    
    overlay = output.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
    cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), WHITE, 1)
    
    # Title
    cv2.putText(output, "SKIN ANALYSIS", (panel_x + 65, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
    
    # === ACNE SECTION ===
    total_acne = len(spots)
    if total_acne == 0:
        sev, col = "Clear", GREEN
    elif total_acne < 5:
        sev, col = "Mild", GREEN
    elif total_acne < 10:
        sev, col = "Moderate", YELLOW
    elif total_acne < 15:
        sev, col = "Notable", ORANGE
    else:
        sev, col = "Severe", RED
    
    cv2.putText(output, f"Acne: {total_acne}  |  {sev}", (panel_x + 15, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.circle(output, (panel_x + panel_w - 25, panel_y + 50), 8, RED, -1)  # Legend
    
    # === BLEMISHES SECTION ===
    total_blemishes = len(blemishes)
    if total_blemishes == 0:
        blem_sev, blem_col = "Clear", GREEN
    elif total_blemishes < 3:
        blem_sev, blem_col = "Few", GREEN
    elif total_blemishes < 7:
        blem_sev, blem_col = "Some", YELLOW
    else:
        blem_sev, blem_col = "Many", ORANGE
    
    cv2.putText(output, f"Blemishes: {total_blemishes}  |  {blem_sev}", (panel_x + 15, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blem_col, 1)
    cv2.circle(output, (panel_x + panel_w - 25, panel_y + 75), 8, (255, 150, 0), -1)  # Blue legend
    
    # === OILINESS SECTION ===
    if oiliness_score < 15:
        oil_sev, oil_col = "Normal", GREEN
    elif oiliness_score < 35:
        oil_sev, oil_col = "Slightly Oily", GREEN
    elif oiliness_score < 55:
        oil_sev, oil_col = "Moderately Oily", YELLOW
    else:
        oil_sev, oil_col = "Very Oily", ORANGE
    
    cv2.putText(output, f"Oiliness: {int(oiliness_score)}%  |  {oil_sev}", (panel_x + 15, panel_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, oil_col, 1)
    cv2.rectangle(output, (panel_x + panel_w - 30, panel_y + 97), (panel_x + panel_w - 15, panel_y + 107), YELLOW, -1)  # Yellow legend
    
    # Region counts
    cv2.line(output, (panel_x + 10, panel_y + 120), (panel_x + panel_w - 10, panel_y + 120), WHITE, 1)
    cv2.putText(output, "ACNE BY REGION:", (panel_x + 15, panel_y + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    
    y_off = panel_y + 165
    for region, count in region_counts.items():
        label = region.replace('_', ' ').title()
        cv2.putText(output, f"{label}: {count}", (panel_x + 20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        y_off += 20
    
    # Legend
    cv2.line(output, (panel_x + 10, y_off + 5), (panel_x + panel_w - 10, y_off + 5), WHITE, 1)
    cv2.putText(output, "LEGEND:", (panel_x + 15, y_off + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
    y_off += 45
    
    cv2.circle(output, (panel_x + 20, y_off), 6, RED, -1)
    cv2.putText(output, "Acne", (panel_x + 35, y_off + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    
    cv2.circle(output, (panel_x + 100, y_off), 6, (255, 150, 0), -1)
    cv2.putText(output, "Blemish", (panel_x + 115, y_off + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    
    cv2.rectangle(output, (panel_x + 190, y_off - 5), (panel_x + 205, y_off + 5), YELLOW, -1)
    cv2.putText(output, "Oily", (panel_x + 212, y_off + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    
    return output


def analyze_image(image_path):
    """Analyze image for skin issues: acne, blemishes, oiliness"""
    print("=" * 50)
    print("   SKIN ANALYSIS v5 - Multi-Feature Detection")
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
    
    # Detect acne (red circles)
    spots = detect_acne_simple(frame, face_mask)
    region_counts = classify_spots(spots, landmarks, h, w)
    
    # Detect blemishes (blue circles) - excludes eyes and nostrils
    blemishes = detect_blemishes(frame, face_mask, landmarks, h, w)
    
    # Detect oiliness (yellow overlay)
    oily_regions, oiliness_score = detect_oiliness(frame, face_mask, landmarks, h, w)
    
    # Calculate product ratios (not shown in UI but output to console)
    ratios = calculate_product_ratios(len(spots), len(blemishes), oiliness_score)
    
    # Draw overlay
    output = draw_overlay(frame, landmarks, spots, blemishes, oily_regions, oiliness_score, region_counts)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"  ACNE SPOTS: {len(spots)}")
    print("-" * 50)
    for region, count in region_counts.items():
        print(f"    {region.replace('_', ' ').title():15} {count}")
    print("-" * 50)
    print(f"  BLEMISHES: {len(blemishes)}")
    print(f"  OILINESS: {int(oiliness_score)}%")
    print(f"  SEVERITY SCORE: {ratios['severity_score']}")
    print("-" * 50)
    print(f"  DISPENSER ({ratios['total_ml']}ml total):")
    print(f"    toner:    {ratios['toner']}% ({ratios['toner'] * ratios['total_ml'] / 100:.2f}ml)")
    print(f"    Treatment:   {ratios['treatment']}% ({ratios['treatment'] * ratios['total_ml'] / 100:.2f}ml)")
    print(f"    Moisturizer: {ratios['moisturizer']}% ({ratios['moisturizer'] * ratios['total_ml'] / 100:.2f}ml)")
    print("=" * 50)
    
    # Save
    base = os.path.splitext(os.path.basename(image_path))[0]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"{base}_result_{ts}.jpg"
    cv2.imwrite(out_file, output)
    print(f"  Saved: {out_file}")
    
    # Show
    cv2.imshow('Skin Analysis Result', output)
    print("  Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'acne_spots': len(spots), 
        'blemishes': len(blemishes),
        'oiliness': oiliness_score,
        'regions': region_counts, 
        'ratios': ratios
    }


def main():
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("=" * 50)
        print("   SKIN ANALYSIS v5")
        print("=" * 50)
        print("\n  Usage: python extract_details.py <image_path>")
        image_path = input("\n  Image path: ").strip().strip('"').strip("'")
    
    if not image_path or not os.path.exists(image_path):
        print("  ERROR: Invalid file path")
        return
    
    analyze_image(image_path)


if __name__ == "__main__":
    main()