import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  TextInput,
  Dimensions,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';

// ⚠️ CHANGE THIS to your computer's local IP address
const DEFAULT_API_URL = 'http://100.67.70.192:8000';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface AnalysisResult {
  acne_count: number;
  acne_severity: string;
  blemish_count: number;
  blemish_severity: string;
  oiliness_score: number;
  oiliness_severity: string;
  severity_score: number;
  regions: Record<string, number>;
  dispenser: {
    cleanser_pct: number;
    treatment_pct: number;
    moisturizer_pct: number;
    total_ml: number;
    cleanser_ml: number;
    treatment_ml: number;
    moisturizer_ml: number;
  };
  result_image: string;
}

export default function SkinAnalysisScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('front');
  const [showCamera, setShowCamera] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const cameraRef = useRef<CameraView>(null);

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.7,
          skipProcessing: false,
        });
        if (photo && photo.base64) {
          console.log('Photo taken, base64 length:', photo.base64.length);
          setCapturedImage(photo.uri);
          setShowCamera(false);
          await analyzeImage(photo.base64);
        } else {
          Alert.alert('Error', 'Failed to capture photo data');
        }
      } catch (err: any) {
        console.error('Camera error:', err);
        Alert.alert('Error', 'Failed to take picture: ' + err.message);
      }
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [3, 4],
        quality: 0.7,
        base64: true,
      });

      console.log('ImagePicker result:', result.canceled ? 'canceled' : 'success');

      if (!result.canceled && result.assets && result.assets[0]) {
        const asset = result.assets[0];
        if (asset.base64) {
          console.log('Image selected, base64 length:', asset.base64.length);
          setCapturedImage(asset.uri);
          await analyzeImage(asset.base64);
        } else {
          Alert.alert('Error', 'Could not get image data. Please try another image.');
        }
      }
    } catch (err: any) {
      console.error('ImagePicker error:', err);
      Alert.alert('Error', 'Failed to pick image: ' + err.message);
    }
  };

  const analyzeImage = async (base64Image: string) => {
    setLoading(true);
    setAnalysisResult(null);
    setResultImage(null);
    setError(null);

    console.log('Starting analysis, sending to:', apiUrl);
    console.log('Base64 length:', base64Image.length);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

      const response = await fetch(`${apiUrl}/analyze-base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      console.log('Response status:', response.status);

      const responseText = await response.text();
      console.log('Response length:', responseText.length);

      if (!response.ok) {
        let errorMsg = 'Analysis failed';
        try {
          const errorData = JSON.parse(responseText);
          errorMsg = errorData.error || errorMsg;
        } catch {
          errorMsg = responseText || errorMsg;
        }
        throw new Error(errorMsg);
      }

      const data: AnalysisResult = JSON.parse(responseText);
      console.log('Analysis complete:', {
        acne: data.acne_count,
        blemishes: data.blemish_count,
        oiliness: data.oiliness_score,
        hasImage: !!data.result_image,
        imageLength: data.result_image?.length || 0,
      });

      setAnalysisResult(data);
      if (data.result_image) {
        setResultImage(`data:image/jpeg;base64,${data.result_image}`);
      }
    } catch (err: any) {
      console.error('Analysis error:', err);
      let errorMessage = 'Could not connect to server';
      if (err.name === 'AbortError') {
        errorMessage = 'Request timed out. Make sure the server is running.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      setError(errorMessage);
      Alert.alert('Analysis Error', errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setCapturedImage(null);
    setResultImage(null);
    setAnalysisResult(null);
    setError(null);
  };

  const testConnection = async () => {
    try {
      const response = await fetch(`${apiUrl}/`, { method: 'GET' });
      const data = await response.json();
      Alert.alert('Connection OK', `Server says: ${data.message}`);
    } catch (err: any) {
      Alert.alert('Connection Failed', `Could not reach ${apiUrl}\n\nError: ${err.message}`);
    }
  };

  // Camera permissions
  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  // Camera view
  if (showCamera) {
    return (
      <View style={styles.cameraContainer}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing={facing}
        />
        {/* Overlay positioned absolutely on top of camera */}
        <View style={styles.cameraOverlayAbsolute}>
          <View style={styles.cameraOverlay}>
            <View style={styles.faceGuide} />
            <Text style={styles.cameraHintText}>Position your face in the oval</Text>
          </View>
          <View style={styles.cameraControls}>
            <TouchableOpacity
              style={styles.cameraButton}
              onPress={() => setShowCamera(false)}
            >
              <Text style={styles.buttonText}>✕</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.cameraButton, styles.captureButton]}
              onPress={takePicture}
            >
              <View style={styles.captureInner} />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.cameraButton}
              onPress={() => setFacing(f => f === 'front' ? 'back' : 'front')}
            >
              <Text style={styles.buttonText}>🔄</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    );
  }

  // Main screen
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <Text style={styles.title}>🧴 Skin Analysis</Text>
      <Text style={styles.subtitle}>AI-Powered Skin Assessment</Text>

      {/* API URL Config */}
      <View style={styles.configSection}>
        <Text style={styles.configLabel}>Server URL:</Text>
        <View style={styles.configRow}>
          <TextInput
            style={styles.configInput}
            value={apiUrl}
            onChangeText={setApiUrl}
            placeholder="http://192.168.1.XXX:8000"
            placeholderTextColor="#666"
            autoCapitalize="none"
            autoCorrect={false}
          />
          <TouchableOpacity style={styles.testButton} onPress={testConnection}>
            <Text style={styles.testButtonText}>Test</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Action Buttons */}
      {!loading && !analysisResult && (
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => {
              if (!permission.granted) {
                requestPermission();
              } else {
                setShowCamera(true);
              }
            }}
          >
            <Text style={styles.actionButtonText}>📷 Take Photo</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.actionButton, styles.secondaryButton]}
            onPress={pickImage}
          >
            <Text style={styles.actionButtonText}>🖼️ Choose from Gallery</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Loading */}
      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4CAF50" />
          <Text style={styles.loadingText}>Analyzing skin...</Text>
          <Text style={styles.loadingSubtext}>This may take a few seconds</Text>
        </View>
      )}

      {/* Error */}
      {error && !loading && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>❌ {error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={resetAnalysis}>
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Results */}
      {analysisResult && (
        <View style={styles.resultsContainer}>
          <Text style={styles.sectionTitle}>✅ Analysis Complete</Text>
          
          {/* Result Image */}
          {resultImage && (
            <View style={styles.imageContainer}>
              <Image 
                source={{ uri: resultImage }} 
                style={styles.resultImage}
                resizeMode="contain"
              />
            </View>
          )}

          {/* Summary Metrics */}
          <View style={styles.metricsGrid}>
            <View style={[styles.metricCard, { borderLeftColor: '#F44336', borderLeftWidth: 4 }]}>
              <Text style={styles.metricValue}>{analysisResult.acne_count}</Text>
              <Text style={styles.metricLabel}>Acne</Text>
              <Text style={[styles.severity, getSeverityStyle(analysisResult.acne_severity)]}>
                {analysisResult.acne_severity}
              </Text>
            </View>
            <View style={[styles.metricCard, { borderLeftColor: '#2196F3', borderLeftWidth: 4 }]}>
              <Text style={styles.metricValue}>{analysisResult.blemish_count}</Text>
              <Text style={styles.metricLabel}>Blemishes</Text>
              <Text style={[styles.severity, getSeverityStyle(analysisResult.blemish_severity)]}>
                {analysisResult.blemish_severity}
              </Text>
            </View>
            <View style={[styles.metricCard, { borderLeftColor: '#FFC107', borderLeftWidth: 4 }]}>
              <Text style={styles.metricValue}>{Math.round(analysisResult.oiliness_score)}%</Text>
              <Text style={styles.metricLabel}>Oiliness</Text>
              <Text style={[styles.severity, getSeverityStyle(analysisResult.oiliness_severity)]}>
                {analysisResult.oiliness_severity}
              </Text>
            </View>
          </View>

          {/* Dispenser Recommendation */}
          <View style={styles.dispenserCard}>
            <Text style={styles.dispenserTitle}>
              💧 Product Recommendation
            </Text>
            <Text style={styles.dispenserSubtitle}>
              Total: {analysisResult.dispenser.total_ml}ml per application
            </Text>
            
            {/* Visual Bar */}
            <View style={styles.dispenserBar}>
              <View style={[styles.barSegment, styles.cleanserBar, { flex: analysisResult.dispenser.cleanser_pct || 1 }]}>
                <Text style={styles.barText}>{analysisResult.dispenser.cleanser_pct}%</Text>
              </View>
              <View style={[styles.barSegment, styles.treatmentBar, { flex: analysisResult.dispenser.treatment_pct || 1 }]}>
                <Text style={styles.barText}>{analysisResult.dispenser.treatment_pct}%</Text>
              </View>
              <View style={[styles.barSegment, styles.moisturizerBar, { flex: analysisResult.dispenser.moisturizer_pct || 1 }]}>
                <Text style={styles.barText}>{analysisResult.dispenser.moisturizer_pct}%</Text>
              </View>
            </View>
            
            {/* Legend with ML amounts */}
            <View style={styles.dispenserDetails}>
              <View style={styles.dispenserRow}>
                <View style={[styles.colorDot, { backgroundColor: '#4CAF50' }]} />
                <Text style={styles.dispenserLabel}>Cleanser</Text>
                <Text style={styles.dispenserValue}>{analysisResult.dispenser.cleanser_ml}ml</Text>
              </View>
              <View style={styles.dispenserRow}>
                <View style={[styles.colorDot, { backgroundColor: '#F44336' }]} />
                <Text style={styles.dispenserLabel}>Treatment</Text>
                <Text style={styles.dispenserValue}>{analysisResult.dispenser.treatment_ml}ml</Text>
              </View>
              <View style={styles.dispenserRow}>
                <View style={[styles.colorDot, { backgroundColor: '#2196F3' }]} />
                <Text style={styles.dispenserLabel}>Moisturizer</Text>
                <Text style={styles.dispenserValue}>{analysisResult.dispenser.moisturizer_ml}ml</Text>
              </View>
            </View>
          </View>

          {/* Region Breakdown */}
          <View style={styles.regionCard}>
            <Text style={styles.regionTitle}>📍 Acne by Region</Text>
            {Object.entries(analysisResult.regions).map(([region, count]) => (
              <View key={region} style={styles.regionRow}>
                <Text style={styles.regionName}>{region.replace(/_/g, ' ')}</Text>
                <View style={styles.regionBar}>
                  <View 
                    style={[
                      styles.regionFill, 
                      { width: `${Math.min(100, (count as number) * 20)}%` }
                    ]} 
                  />
                </View>
                <Text style={styles.regionCount}>{count}</Text>
              </View>
            ))}
          </View>

          {/* Reset Button */}
          <TouchableOpacity style={styles.resetButton} onPress={resetAnalysis}>
            <Text style={styles.resetButtonText}>🔄 New Analysis</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

const getSeverityStyle = (severity: string) => {
  const lower = severity.toLowerCase();
  if (lower.includes('clear') || lower.includes('normal') || lower.includes('mild') || lower.includes('few')) {
    return { color: '#4CAF50' };
  } else if (lower.includes('moderate') || lower.includes('some') || lower.includes('slightly')) {
    return { color: '#FFC107' };
  } else {
    return { color: '#F44336' };
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  scrollContent: {
    padding: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
    marginBottom: 24,
  },
  configSection: {
    marginBottom: 24,
  },
  configLabel: {
    color: '#888',
    marginBottom: 8,
    fontSize: 14,
  },
  configRow: {
    flexDirection: 'row',
    gap: 10,
  },
  configInput: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    color: '#fff',
    padding: 14,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#333',
    fontSize: 14,
  },
  testButton: {
    backgroundColor: '#333',
    paddingHorizontal: 20,
    borderRadius: 10,
    justifyContent: 'center',
  },
  testButtonText: {
    color: '#4CAF50',
    fontWeight: '600',
  },
  buttonContainer: {
    gap: 14,
  },
  actionButton: {
    backgroundColor: '#4CAF50',
    padding: 20,
    borderRadius: 14,
    alignItems: 'center',
  },
  secondaryButton: {
    backgroundColor: '#2196F3',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  cameraOverlayAbsolute: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  faceGuide: {
    width: 260,
    height: 340,
    borderWidth: 3,
    borderColor: 'rgba(76, 175, 80, 0.7)',
    borderRadius: 130,
  },
  cameraHintText: {
    color: '#fff',
    marginTop: 20,
    fontSize: 16,
    textShadow: '1px 1px 3px #000',
  },
  cameraControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: 30,
    paddingBottom: 50,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  cameraButton: {
    width: 60,
    height: 60,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    backgroundColor: '#fff',
    width: 80,
    height: 80,
    borderRadius: 40,
    padding: 4,
  },
  captureInner: {
    flex: 1,
    backgroundColor: '#4CAF50',
    borderRadius: 36,
  },
  buttonText: {
    color: '#fff',
    fontSize: 24,
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 60,
  },
  loadingText: {
    color: '#fff',
    marginTop: 20,
    fontSize: 18,
    fontWeight: '600',
  },
  loadingSubtext: {
    color: '#666',
    marginTop: 8,
    fontSize: 14,
  },
  errorContainer: {
    backgroundColor: '#1a1a1a',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#F44336',
  },
  errorText: {
    color: '#F44336',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 15,
  },
  retryButton: {
    backgroundColor: '#F44336',
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  resultsContainer: {
    marginTop: 10,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginBottom: 20,
    textAlign: 'center',
  },
  imageContainer: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 10,
    marginBottom: 20,
  },
  resultImage: {
    width: '100%',
    height: SCREEN_WIDTH - 60,
    borderRadius: 12,
  },
  metricsGrid: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },
  metricCard: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  metricValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
  },
  metricLabel: {
    fontSize: 13,
    color: '#888',
    marginTop: 4,
  },
  severity: {
    fontSize: 13,
    fontWeight: '600',
    marginTop: 6,
  },
  dispenserCard: {
    backgroundColor: '#1a1a1a',
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
  },
  dispenserTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 4,
  },
  dispenserSubtitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 16,
  },
  dispenserBar: {
    flexDirection: 'row',
    height: 36,
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 16,
  },
  barSegment: {
    justifyContent: 'center',
    alignItems: 'center',
    minWidth: 40,
  },
  barText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  cleanserBar: {
    backgroundColor: '#4CAF50',
  },
  treatmentBar: {
    backgroundColor: '#F44336',
  },
  moisturizerBar: {
    backgroundColor: '#2196F3',
  },
  dispenserDetails: {
    gap: 12,
  },
  dispenserRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  colorDot: {
    width: 14,
    height: 14,
    borderRadius: 7,
    marginRight: 12,
  },
  dispenserLabel: {
    flex: 1,
    color: '#ccc',
    fontSize: 15,
  },
  dispenserValue: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  regionCard: {
    backgroundColor: '#1a1a1a',
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
  },
  regionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 16,
  },
  regionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  regionName: {
    color: '#ccc',
    textTransform: 'capitalize',
    width: 100,
    fontSize: 14,
  },
  regionBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#333',
    borderRadius: 4,
    marginHorizontal: 12,
    overflow: 'hidden',
  },
  regionFill: {
    height: '100%',
    backgroundColor: '#F44336',
    borderRadius: 4,
  },
  regionCount: {
    color: '#fff',
    fontWeight: '700',
    width: 30,
    textAlign: 'right',
  },
  resetButton: {
    backgroundColor: '#333',
    padding: 18,
    borderRadius: 14,
    alignItems: 'center',
    marginTop: 10,
  },
  resetButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '600',
  },
});
