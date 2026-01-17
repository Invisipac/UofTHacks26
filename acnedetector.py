import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import sys

from analyticsmeasure import track_skin_analysis

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
# MediaPipe Face Mesh setup
# -----------------------------
from mediapipe.tasks import python
from mediapipe. tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker. create_from_options(options)

# -----------------------------
# Debug version - shows all intermediate steps
# -----------------------------
def detect_acne_debug(image, landmarks, output_dir="acne_output"):
    """
    Debug version that saves all intermediate detection steps
    """
    h, w, _ = image.shape
    
    print("\n" + "="*70)
    print("DEBUG MODE - Analyzing detection steps...")
    print("="*70)
    
    # Define face skin area
    FACE_SKIN = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    # Create face mask
    points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in FACE_SKIN
    ])
    
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [points], 255)
    
    # Save face mask
    cv2.imwrite(os.path. join(output_dir, "debug_1_face_mask.jpg"), face_mask)
    print("✓ Saved face mask")
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # ===== METHOD 1: HSV =====
    print("\n--- METHOD 1: HSV Redness Detection ---")
    lower_red1 = np.array([0, 15, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 15, 40])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_spots_hsv = cv2.bitwise_or(red_mask1, red_mask2)
    red_spots_hsv_masked = cv2.bitwise_and(red_spots_hsv, face_mask)
    
    pixel_count_hsv = cv2.countNonZero(red_spots_hsv_masked)
    print(f"HSV detected pixels: {pixel_count_hsv}")
    cv2.imwrite(os.path.join(output_dir, "debug_2_hsv_detection.jpg"), red_spots_hsv_masked)
    
    # ===== METHOD 2: LAB =====
    print("\n--- METHOD 2: LAB 'a' Channel Detection ---")
    a_channel = lab[:,: ,1]
    
    face_skin_pixels = a_channel[face_mask > 0]
    mean_a = np.mean(face_skin_pixels)
    std_a = np.std(face_skin_pixels)
    
    threshold_a = mean_a + 0.2 * std_a
    print(f"LAB 'a' channel:  mean={mean_a:.2f}, std={std_a:.2f}, threshold={threshold_a:.2f}")
    
    red_spots_lab = np.zeros_like(a_channel, dtype=np.uint8)
    red_spots_lab[a_channel > threshold_a] = 255
    red_spots_lab_masked = cv2.bitwise_and(red_spots_lab, face_mask)
    
    pixel_count_lab = cv2.countNonZero(red_spots_lab_masked)
    print(f"LAB detected pixels: {pixel_count_lab}")
    cv2.imwrite(os.path.join(output_dir, "debug_3_lab_detection.jpg"), red_spots_lab_masked)
    
    # ===== METHOD 3: YCrCb =====
    print("\n--- METHOD 3: YCrCb Cr Channel Detection ---")
    cr_channel = ycrcb[:,: ,1]
    
    face_cr_pixels = cr_channel[face_mask > 0]
    mean_cr = np.mean(face_cr_pixels)
    std_cr = np.std(face_cr_pixels)
    
    threshold_cr = mean_cr + 0.3 * std_cr
    print(f"YCrCb Cr channel: mean={mean_cr:.2f}, std={std_cr:.2f}, threshold={threshold_cr:.2f}")
    
    red_spots_ycrcb = np.zeros_like(cr_channel, dtype=np.uint8)
    red_spots_ycrcb[cr_channel > threshold_cr] = 255
    red_spots_ycrcb_masked = cv2.bitwise_and(red_spots_ycrcb, face_mask)
    
    pixel_count_ycrcb = cv2.countNonZero(red_spots_ycrcb_masked)
    print(f"YCrCb detected pixels: {pixel_count_ycrcb}")
    cv2.imwrite(os. path.join(output_dir, "debug_4_ycrcb_detection.jpg"), red_spots_ycrcb_masked)
    
    # ===== COMBINE ALL =====
    print("\n--- Combining All Methods ---")
    combined = cv2.bitwise_or(red_spots_hsv_masked, red_spots_lab_masked)
    combined = cv2.bitwise_or(combined, red_spots_ycrcb_masked)
    
    pixel_count_combined = cv2.countNonZero(combined)
    print(f"Combined detected pixels: {pixel_count_combined}")
    cv2.imwrite(os.path. join(output_dir, "debug_5_combined.jpg"), combined)
    
    # ===== NOISE REMOVAL =====
    print("\n--- Noise Removal ---")
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    pixel_count_after_open = cv2.countNonZero(cleaned)
    print(f"After MORPH_OPEN: {pixel_count_after_open} pixels")
    cv2.imwrite(os.path.join(output_dir, "debug_6_after_open.jpg"), cleaned)
    
    kernel2 = np.ones((4, 4), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2, iterations=1)
    
    pixel_count_after_close = cv2.countNonZero(cleaned)
    print(f"After MORPH_CLOSE: {pixel_count_after_close} pixels")
    cv2.imwrite(os.path.join(output_dir, "debug_7_after_close.jpg"), cleaned)
    
    cleaned = cv2.medianBlur(cleaned, 3)
    
    pixel_count_after_blur = cv2.countNonZero(cleaned)
    print(f"After median blur: {pixel_count_after_blur} pixels")
    cv2.imwrite(os.path.join(output_dir, "debug_8_after_blur.jpg"), cleaned)
    
    # ===== FIND CONTOURS =====
    print("\n--- Finding Contours ---")
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Total contours found: {len(contours)}")
    
    # Draw all contours for debugging
    all_contours_img = image.copy()
    cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "debug_9_all_contours.jpg"), all_contours_img)
    
    # Filter by size and circularity
    acne_spots = 0
    acne_locations = []
    highlighted_image = image.copy()
    final_mask = np.zeros_like(cleaned)
    
    print("\nFiltering contours by size and circularity:")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area > 0:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Very permissive filtering
            if 10 < area < 5000 and circularity > 0.1:
                acne_spots += 1
                
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                radius = max(radius, 4)
                
                acne_locations.append({
                    'center': center,
                    'radius': radius,
                    'area': area,
                    'circularity': circularity
                })
                
                cv2.drawContours(final_mask, [contour], -1, 255, -1)
                cv2.circle(highlighted_image, center, radius + 3, (0, 0, 255), 2)
                cv2.circle(highlighted_image, center, 2, (0, 0, 255), -1)
                
                if i < 10: 
                    print(f"  Contour {i+1}: area={area:. 0f}, circularity={circularity:.2f}, center={center}")
    
    print(f"\nTotal spots after filtering: {acne_spots}")
    
    cv2.imwrite(os.path.join(output_dir, "debug_10_final_mask.jpg"), final_mask)
    cv2.imwrite(os.path.join(output_dir, "debug_11_highlighted.jpg"), highlighted_image)
    
    # Determine if acne is present
    has_acne = acne_spots > 3
    
    if acne_spots == 0:
        severity = "None"
        confidence = "High"
    elif acne_spots <= 3:
        severity = "Minimal"
        confidence = "Medium"
    elif acne_spots <= 12:
        severity = "Mild"
        confidence = "High"
    elif acne_spots <= 30:
        severity = "Moderate"
        confidence = "High"
    else:
        severity = "Severe"
        confidence = "High"
    
    print("\n" + "="*70)
    print("DEBUG SUMMARY")
    print("="*70)
    print(f"Pixels detected by HSV: {pixel_count_hsv}")
    print(f"Pixels detected by LAB: {pixel_count_lab}")
    print(f"Pixels detected by YCrCb: {pixel_count_ycrcb}")
    print(f"Combined pixels: {pixel_count_combined}")
    print(f"After noise removal: {pixel_count_after_blur}")
    print(f"Total contours: {len(contours)}")
    print(f"Valid acne spots: {acne_spots}")
    print("="*70 + "\n")
    
    return has_acne, acne_spots, severity, confidence, highlighted_image, acne_locations, final_mask


def create_result_image(image, has_acne, spot_count, severity, confidence):
    """Create an image with text overlay showing results"""
    result_image = image.copy()
    
    if has_acne:
        result_text = "ACNE DETECTED"
        result_color = (0, 0, 255)
        box_color = (0, 0, 255)
    else:
        result_text = "NO ACNE DETECTED"
        result_color = (0, 255, 0)
        box_color = (0, 255, 0)
    
    cv2.rectangle(result_image, (20, 20), (480, 200), box_color, 3)
    
    overlay = result_image.copy()
    cv2.rectangle(overlay, (20, 20), (480, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, result_image, 0.5, 0, result_image)
    cv2.rectangle(result_image, (20, 20), (480, 200), box_color, 3)
    
    cv2.putText(result_image, result_text, (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, result_color, 3)
    
    cv2.putText(result_image, f"Spots detected: {spot_count}", (40, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(result_image, f"Severity: {severity}", (40, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(result_image, f"Confidence: {confidence}", (40, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result_image


def create_combined_view(original, highlighted, result, mask):
    """Create a 2x2 grid showing all views"""
    h, w = original.shape[:2]
    
    target_size = (w // 2, h // 2)
    
    original_resized = cv2.resize(original, target_size)
    highlighted_resized = cv2.resize(highlighted, target_size)
    result_resized = cv2.resize(result, target_size)
    
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask_bgr, target_size)
    
    cv2.putText(original_resized, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(highlighted_resized, "Highlighted Acne", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_resized, "Analysis Result", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(mask_resized, "Detection Mask", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    top_row = np.hstack([original_resized, highlighted_resized])
    bottom_row = np.hstack([result_resized, mask_resized])
    combined = np.vstack([top_row, bottom_row])
    
    return combined


# -----------------------------
# Main analysis function
# -----------------------------
def analyze_image(image_path, output_dir="acne_output"):
    """
    Analyze a static image for acne detection with debug output
    """
    
    print("\n" + "="*70)
    print("ACNE DETECTION - DEBUG MODE")
    print("="*70)
    print(f"Analyzing: {image_path}")
    print("="*70 + "\n")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return None
    
    print(f"Image loaded:  {image. shape[1]}x{image.shape[0]} pixels")
    
    os.makedirs(output_dir, exist_ok=True)
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat. SRGB, data=rgb)
    
    print("Detecting face landmarks...")
    detection_result = detector.detect(mp_image)
    
    if not detection_result. face_landmarks:
        print("Error: No face detected in the image!")
        return None
    
    print("Face detected successfully!")
    landmarks = detection_result.face_landmarks[0]
    
    has_acne, spot_count, severity, confidence, highlighted_img, acne_locs, acne_mask = detect_acne_debug(
        image, landmarks, output_dir
    )
    
    result_image = create_result_image(image, has_acne, spot_count, severity, confidence)
    result_highlighted = create_result_image(highlighted_img, has_acne, spot_count, severity, confidence)
    combined_view = create_combined_view(image, highlighted_img, result_highlighted, acne_mask)
    
    print("\n" + "="*70)
    print("FINAL ANALYSIS RESULTS")
    print("="*70)
    
    if has_acne:
        print("Result: ACNE DETECTED ❌")
    else:
        print("Result:  NO ACNE DETECTED ✓")
    
    print(f"Spots detected: {spot_count}")
    print(f"Severity: {severity}")
    print(f"Confidence:  {confidence}")
    print("="*70)
    
    if acne_locs:
        print(f"\nDetected acne locations:")
        for i, loc in enumerate(acne_locs[: 20], 1):
            print(f"  {i: 2d}. Center: {loc['center']}, "
                  f"Radius:  {loc['radius']: 2d}px, "
                  f"Area: {loc['area']:5. 0f}px², "
                  f"Circularity: {loc['circularity']:.2f}")
        if len(acne_locs) > 20:
            print(f"  ... and {len(acne_locs) - 20} more")
    else:
        print("\nNo acne spots detected.")
    
    print(f"\nSaving output files to '{output_dir}/'...")
    cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), image)
    cv2.imwrite(os.path.join(output_dir, "2_highlighted.jpg"), highlighted_img)
    cv2.imwrite(os.path. join(output_dir, "3_result.jpg"), result_highlighted)
    cv2.imwrite(os.path.join(output_dir, "4_mask.jpg"), acne_mask)
    cv2.imwrite(os.path.join(output_dir, "5_combined_view.jpg"), combined_view)
    
    print("\nDebug files saved (check these to see what's being detected):")
    print("  - debug_1_face_mask.jpg")
    print("  - debug_2_hsv_detection. jpg")
    print("  - debug_3_lab_detection. jpg")
    print("  - debug_4_ycrcb_detection.jpg")
    print("  - debug_5_combined.jpg")
    print("  - debug_6_after_open. jpg")
    print("  - debug_7_after_close. jpg")
    print("  - debug_8_after_blur. jpg")
    print("  - debug_9_all_contours.jpg")
    print("  - debug_10_final_mask.jpg")
    print("  - debug_11_highlighted.jpg")
    print("="*70 + "\n")
    
    print("Displaying results...  Press any key to close.")
    
    cv2.imshow("Acne Analysis - Combined View", combined_view)
    cv2.imshow("Highlighted Acne Spots", highlighted_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'has_acne': has_acne,
        'spot_count': spot_count,
        'severity': severity,
        'confidence': confidence,
        'acne_locations': acne_locs
    }


# -----------------------------
# Command-line interface
# -----------------------------
if __name__ == "__main__": 
    print("\n" + "="*70)
    print("ACNE DETECTOR - DEBUG MODE WITH AMPLITUDE")
    print("="*70)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:  
        print("\nNo image path provided. Enter image path:")
        image_path = input("Image path: ").strip()
    
    if not image_path:
        print("\nUsage: python acne_detector_debug.py <image_path>")
        print("Example: python acne_detector_debug.py face.jpg")
        sys.exit(1)
    
    output_dir = "acne_output"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Run analysis
    results = analyze_image(image_path, output_dir)
    
    if results:
        print("\n✓ Analysis complete!  Check debug files in output folder.")
        
        # Send to Amplitude
        print("\n" + "="*70)
        print("SENDING TO AMPLITUDE")
        print("="*70)
        
        try:
            from analyticsmeasure import track_skin_analysis
            
            acne_data = {
                'has_acne': results['has_acne'],
                'spot_count': results['spot_count'],
                'severity':  results['severity'],
                'confidence': results['confidence'],
                'acne_locations': results. get('acne_locations', [])
            }
            
            skin_data = results.get('skin_metrics')  # Will be None in debug version
            
            print(f"\nPreparing to send data for user: invisiblepacman")
            print(f"  Has Acne: {acne_data['has_acne']}")
            print(f"  Spot Count: {acne_data['spot_count']}")
            print(f"  Severity: {acne_data['severity']}")
            
            # Send to Amplitude
            success = track_skin_analysis("invisiblepacman", acne_data, skin_data)
            
            if success:
                print("\n✓ Data successfully sent to Amplitude!")
                print("\nTo view in Amplitude:")
                print("1. Go to https://analytics.amplitude.com")
                print("2. Click 'Users' → 'User Look-Up'")
                print("3. Search for: invisiblepacman")
                print("4. Look for 'Skin Analysis Completed' event")
            else:
                print("\n❌ Failed to send to Amplitude")
                
        except ImportError as e:
            print(f"\n❌ Error:  Could not import analyticsmeasure")
            print(f"Make sure analyticsmeasure.py is in the same folder")
            print(f"Error details: {e}")
        except Exception as e:
            print(f"\n❌ Error sending to Amplitude: {e}")
        
        print("="*70)
        
    else:
        print("\n✗ Analysis failed.")