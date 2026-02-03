from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import tempfile
import os

# Import the CV analysis functions from extract_details
from extract_details import (
    detector, create_face_mask, detect_acne_simple, detect_blemishes,
    detect_oiliness, classify_spots, calculate_product_ratios, draw_overlay,
    get_polygon, FACE_OUTLINE
)
import mediapipe as mp

app = FastAPI(title="Skin Analysis API")

# Allow CORS for Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def analyze_image_api(image_bytes):
    """Analyze image and return results with annotated image"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return None, "Could not decode image"
    
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
        return None, "No face detected"
    
    landmarks = result.face_landmarks[0]
    face_mask = create_face_mask(landmarks, h, w)
    
    # Run all detections
    spots = detect_acne_simple(frame, face_mask)
    region_counts = classify_spots(spots, landmarks, h, w)
    blemishes = detect_blemishes(frame, face_mask, landmarks, h, w)
    oily_regions, oiliness_score = detect_oiliness(frame, face_mask, landmarks, h, w)
    
    # Calculate ratios
    ratios = calculate_product_ratios(len(spots), len(blemishes), oiliness_score)
    
    # Draw overlay
    output = draw_overlay(frame, landmarks, spots, blemishes, oily_regions, oiliness_score, region_counts)
    
    # Convert output image to base64
    _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Determine severity labels
    acne_severity = "Clear" if len(spots) == 0 else "Mild" if len(spots) < 5 else "Moderate" if len(spots) < 10 else "Notable" if len(spots) < 15 else "Severe"
    blemish_severity = "Clear" if len(blemishes) == 0 else "Few" if len(blemishes) < 3 else "Some" if len(blemishes) < 7 else "Many"
    oil_severity = "Normal" if oiliness_score < 15 else "Slightly Oily" if oiliness_score < 35 else "Moderately Oily" if oiliness_score < 55 else "Very Oily"
    
    return {
        "acne_count": len(spots),
        "acne_severity": acne_severity,
        "blemish_count": len(blemishes),
        "blemish_severity": blemish_severity,
        "oiliness_score": round(oiliness_score, 1),
        "oiliness_severity": oil_severity,
        "severity_score": ratios["severity_score"],
        "regions": region_counts,
        "dispenser": {
            "toner_pct": ratios["toner"],
            "treatment_pct": ratios["treatment"],
            "moisturizer_pct": ratios["moisturizer"],
            "total_ml": ratios["total_ml"],
            "toner_ml": round(ratios["toner"] * ratios["total_ml"] / 100, 2),
            "treatment_ml": round(ratios["treatment"] * ratios["total_ml"] / 100, 2),
            "moisturizer_ml": round(ratios["moisturizer"] * ratios["total_ml"] / 100, 2),
        },
        "result_image": img_base64
    }, None


@app.get("/")
async def root():
    return {"message": "Skin Analysis API", "status": "running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded image for skin issues"""
    try:
        contents = await file.read()
        result, error = analyze_image_api(contents)
        
        if error:
            return JSONResponse(status_code=400, content={"error": error})
        
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze-base64")
async def analyze_base64(data: dict):
    """Analyze base64 encoded image"""
    try:
        # Remove data URL prefix if present
        img_data = data.get("image", "")
        if not img_data:
            return JSONResponse(status_code=400, content={"error": "No image data provided"})
        
        if "," in img_data:
            img_data = img_data.split(",")[1]
        
        print(f"  Received image data, length: {len(img_data)}")
        
        image_bytes = base64.b64decode(img_data)
        print(f"  Decoded to {len(image_bytes)} bytes")
        
        result, error = analyze_image_api(image_bytes)
        
        if error:
            print(f"  Analysis error: {error}")
            return JSONResponse(status_code=400, content={"error": error})
        
        print(f"  Analysis complete: {result['acne_count']} acne, {result['blemish_count']} blemishes")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("   SKIN ANALYSIS API SERVER")
    print("=" * 50)
    print("  Starting server on http://0.0.0.0:8000")
    print("  Find your computer's IP address and use that in the Expo app")
    print("  Example: http://192.168.1.XXX:8000")
    print("=" * 50)
    # 0.0.0.0 allows your phone to find your computer on Wi-Fi
    uvicorn.run(app, host="0.0.0.0", port=8000)