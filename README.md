# 🧴 Skin Analysis App

AI-powered skin analysis tool that detects acne, blemishes, and oiliness using computer vision, then recommends personalized skincare product ratios.

## 📋 Prerequisites

- **Python 3.10+** installed
- **Node.js 18+** installed
- **Expo Go** app on your phone ([iOS](https://apps.apple.com/app/expo-go/id982107779) / [Android](https://play.google.com/store/apps/details?id=host.exp.exponent))
- Phone and computer on the **same WiFi network**

## 🚀 Quick Start

### Step 1: Install Python Dependencies

```bash
cd c:\Users\ilike\Desktop\UofTHacks26
pip install fastapi uvicorn python-multipart opencv-python mediapipe numpy
```

### Step 2: Install Expo Dependencies

```bash
cd c:\Users\ilike\Desktop\UofTHacks26\uofthacks26
npm install
npx expo install expo-camera expo-image-picker
```

### Step 3: Find Your Computer's IP Address

```bash
# Windows
ipconfig | findstr "IPv4"

# Mac/Linux
ifconfig | grep "inet "
```

Note your IP (e.g., `192.168.1.100` or `100.67.70.192`)

### Step 4: Update the App with Your IP

Edit `uofthacks26/app/(tabs)/index.tsx` line ~17:
```typescript
const DEFAULT_API_URL = 'http://YOUR_IP_HERE:8000';
```

Or just change it in the app's text field when running.

### Step 5: Start the Python Server

```bash
cd c:\Users\ilike\Desktop\UofTHacks26
python main.py
```

You should see:
```
==================================================
   SKIN ANALYSIS API SERVER
==================================================
  Starting server on http://0.0.0.0:8000
...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Start the Expo App

In a **new terminal**:
```bash
cd c:\Users\ilike\Desktop\UofTHacks26\uofthacks26
npx expo start
```

### Step 7: Open on Your Phone

1. Open **Expo Go** app on your phone
2. Scan the QR code shown in the terminal
3. The app will load on your phone

## 📱 Using the App

1. **Test Connection**: Tap "Test" button to verify server connection
2. **Take Photo**: Use camera to capture your face
3. **Choose Photo**: Or select from gallery
4. **View Results**: See analysis with:
   - Annotated image showing detected spots
   - Acne count & severity
   - Blemish count & severity
   - Oiliness percentage
   - Personalized product ratios (cleanser/treatment/moisturizer)

## 🔧 Troubleshooting

### "Connection Refused" Error
- Make sure Python server is running (`python main.py`)
- Check that IP address is correct in the app
- Ensure phone and computer are on same WiFi
- Try disabling Windows Firewall temporarily

### "No Face Detected" Error
- Ensure good lighting
- Face the camera directly
- Keep face within the oval guide

### Server Won't Start
```bash
# Install missing dependencies
pip install fastapi uvicorn python-multipart opencv-python mediapipe numpy
```

### Expo App Won't Load
```bash
# Clear cache and restart
npx expo start --clear
```

## 📁 Project Structure

```
UofTHacks26/
├── main.py              # FastAPI server
├── extract_details.py   # CV analysis logic
├── face_landmarker.task # MediaPipe model (auto-downloads)
├── README.md            # This file
└── uofthacks26/         # Expo mobile app
    ├── app/
    │   └── (tabs)/
    │       └── index.tsx  # Main app screen
    └── package.json
```

## 🧪 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/analyze` | POST | Analyze uploaded image file |
| `/analyze-base64` | POST | Analyze base64 encoded image |

## 📊 Analysis Output

```json
{
  "acne_count": 5,
  "acne_severity": "Mild",
  "blemish_count": 2,
  "blemish_severity": "Few",
  "oiliness_score": 25.5,
  "oiliness_severity": "Slightly Oily",
  "dispenser": {
    "cleanser_pct": 30,
    "treatment_pct": 35,
    "moisturizer_pct": 35,
    "total_ml": 3.5
  },
  "result_image": "<base64 encoded annotated image>"
}
```

## 🎨 Visual Indicators

- 🔴 **Red circles**: Acne spots
- 🔵 **Blue circles**: Other blemishes
- 🟡 **Yellow overlay**: Oily areas (T-zone)

---

Built at **UofT Hacks 2026** 🏆
