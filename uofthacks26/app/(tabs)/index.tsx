import React, { useState, useRef, useEffect } from 'react';
import { Platform, View, Text } from 'react-native';
import { Stack } from 'expo-router';

// --- Types ---
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

type ViewState = 'landing' | 'camera' | 'analyzing' | 'results' | 'error';

// --- Web Style Injector (Makes Tailwind work in Expo Web) ---
const WebStyleInjector = () => {
  useEffect(() => {
    if (Platform.OS === 'web') {
      // 1. Inject Tailwind CDN
      if (!document.getElementById('tailwind-script')) {
        const script = document.createElement('script');
        script.id = 'tailwind-script';
        script.src = "https://cdn.tailwindcss.com";
        script.async = true;
        script.onload = () => {
           // @ts-ignore
           window.tailwind.config = {
              theme: {
                extend: {
                  colors: {
                    cream: '#FAF9F6', 
                    pastel: '#BCE2BF', // The pastel green accent
                    'pastel-glow': '#4ade80', // Brighter for the glow effect
                    'forest-bg': '#1a2e1f', // Deep Forest Green Background
                    'card-glass': 'rgba(255, 255, 255, 0.05)',
                  },
                  fontFamily: {
                    sans: ['Inter', 'sans-serif'],
                  },
                  animation: {
                    'fade-in-up': 'fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                    'fade-in': 'fadeIn 0.5s ease-out forwards',
                    'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    'float': 'float 6s ease-in-out infinite',
                    'scan': 'scan 3s linear infinite',
                  },
                  keyframes: {
                    fadeInUp: {
                      '0%': { opacity: '0', transform: 'translateY(20px)' },
                      '100%': { opacity: '1', transform: 'translateY(0)' },
                    },
                    fadeIn: {
                      '0%': { opacity: '0' },
                      '100%': { opacity: '1' },
                    },
                    float: {
                      '0%, 100%': { transform: 'translateY(0)' },
                      '50%': { transform: 'translateY(-10px)' },
                    },
                    scan: {
                      '0%': { top: '0%' },
                      '100%': { top: '100%' },
                    }
                  },
                },
              },
           };
        };
        document.head.appendChild(script);
      }

      // 2. Inject Fonts (Inter)
      if (!document.getElementById('google-fonts')) {
        const link = document.createElement('link');
        link.id = 'google-fonts';
        link.href = "https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&display=swap";
        link.rel = "stylesheet";
        document.head.appendChild(link);
      }

      // 3. Inject Custom CSS
      if (!document.getElementById('custom-styles')) {
        const style = document.createElement('style');
        style.id = 'custom-styles';
        style.textContent = `
          body { background-color: #1a2e1f; color: #FFFFFF; font-family: 'Inter', sans-serif; overflow-x: hidden; overflow-y: auto; }
          /* Allow vertical scrolling */
          html, body { height: 100%; min-height: 100%; }
          #root { min-height: 100vh; display: flex; flex-direction: column; }
          
          ::-webkit-scrollbar { width: 8px; }
          ::-webkit-scrollbar-track { background: #1a2e1f; }
          ::-webkit-scrollbar-thumb { background: #BCE2BF; border-radius: 4px; }
          
          .glass-card { 
            background: rgba(255, 255, 255, 0.05); 
            backdrop-filter: blur(16px); 
            -webkit-backdrop-filter: blur(16px); 
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
          }
          
          .text-glow {
            text-shadow: 0 0 15px rgba(188, 226, 191, 0.5);
          }
          
          .text-glow-strong {
            text-shadow: 0 0 20px rgba(188, 226, 191, 0.8), 0 0 10px rgba(188, 226, 191, 0.4);
          }
        `;
        document.head.appendChild(style);
      }
    }
  }, []);
  return null;
};

// --- Components ---

// Button Component
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline';
  className?: string;
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({ variant = 'primary', className = '', children, ...props }) => {
  const baseStyles = "relative inline-flex items-center justify-center px-8 py-4 rounded-full font-bold tracking-wide transition-all duration-300 transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed group overflow-hidden uppercase text-sm";
  
  let variantStyles = "";
  
  switch (variant) {
    case 'primary':
      // Pastel Green Button
      variantStyles = "bg-pastel text-forest-bg shadow-[0_0_20px_rgba(188,226,191,0.4)] hover:bg-white hover:shadow-[0_0_30px_rgba(188,226,191,0.6)] border border-pastel";
      break;
    case 'secondary':
      variantStyles = "bg-white/10 text-white border border-white/20 hover:bg-white/20 backdrop-blur-md";
      break;
    case 'outline':
      variantStyles = "bg-transparent text-gray-300 border border-gray-600 hover:border-pastel hover:text-pastel hover:shadow-[0_0_15px_rgba(188,226,191,0.3)]";
      break;
  }

  return (
    // @ts-ignore: Standard HTML elements in Expo Web
    <button className={`${baseStyles} ${variantStyles} ${className}`} {...props}>
      {variant === 'primary' && (
        // @ts-ignore
        <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/40 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700 ease-in-out" />
      )}
      {/* @ts-ignore */}
      <span className="relative flex items-center gap-2 z-10">{children}</span>
    </button>
  );
};

// AnalysisResultView Component
const getSeverityColor = (severity: string) => {
  const s = severity.toLowerCase();
  if (s.includes('clear') || s.includes('normal') || s.includes('mild') || s.includes('few')) {
    return 'text-emerald-400';
  } else if (s.includes('moderate') || s.includes('some') || s.includes('slightly')) {
    return 'text-yellow-400';
  }
  return 'text-rose-400';
};

const AnalysisResultView: React.FC<{ result: AnalysisResult; onReset: () => void }> = ({ result, onReset }) => {
  const [isPrinting, setIsPrinting] = useState(false);

  const handlePrintFormula = async () => {
    setIsPrinting(true);
    try {
      // Convert percentages to milliseconds (100% = 10 seconds = 10000 ms)
      // Since percentages are 0-100 range, multiply by 100 to get ms (1% = 100ms)
      const t1 = Math.round(result.dispenser.cleanser_pct * 200);
      const t2 = Math.round(result.dispenser.treatment_pct * 200);
      const t3 = Math.round(result.dispenser.moisturizer_pct * 200);

      const url = `http://${ESP32_IP}/dispense?t1=${t1}&t2=${t2}&t3=${t3}`;
      console.log('Sending formula to ESP32:', url);

      // Try with no-cors mode if regular fetch fails (ESP32 may not support CORS)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      try {
        await fetch(url, {
          method: 'GET',
          mode: 'no-cors', // Bypass CORS - ESP32 likely doesn't send CORS headers
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        // With no-cors mode, we can't read the response, but the request was sent
        alert('Formula sent successfully to dispenser!');
      } catch (fetchError: any) {
        clearTimeout(timeoutId);
        // If no-cors also fails, try regular fetch as fallback
        if (fetchError.name === 'AbortError') {
          throw new Error('Request timeout - ESP32 may be unreachable');
        }
        throw fetchError;
      }
    } catch (error: any) {
      console.error('Error printing formula:', error);
      const errorMsg = error.message || 'ESP32 unreachable';
      
      // More helpful error messages
      if (error.message?.includes('Failed to fetch') || error.message?.includes('NetworkError')) {
        alert(`Cannot connect to ESP32 at ${ESP32_IP}.\n\nPlease check:\n- ESP32 is powered on\n- ESP32 is on the same network\n- IP address is correct`);
      } else {
        alert(`Failed to send formula: ${errorMsg}`);
      }
    } finally {
      setIsPrinting(false);
    }
  };

  return (
    // @ts-ignore
    <div className="w-full max-w-7xl mx-auto space-y-8 lg:space-y-12 animate-fade-in-up pb-24 px-4 font-sans">
      
      {/* Header Badge */}
      {/* @ts-ignore */}
      <div className="flex justify-center mb-4 lg:mb-8">
        {/* @ts-ignore */}
        <div className="inline-flex items-center gap-2 px-6 py-2 rounded-full bg-pastel/10 border border-pastel/30 text-pastel text-sm font-bold tracking-widest uppercase shadow-[0_0_15px_rgba(188,226,191,0.2)]">
          {/* @ts-ignore */}
          <span className="w-2 h-2 rounded-full bg-pastel animate-pulse" />
          Scan Complete
        </div>
      </div>

      {/* @ts-ignore */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Image (Larger focus) */}
        {/* @ts-ignore */}
        <div className="lg:col-span-7 relative group">
          {/* @ts-ignore */}
          <div className="absolute -inset-2 bg-pastel rounded-[2rem] blur-xl opacity-20 group-hover:opacity-40 transition duration-1000"></div>
          {/* @ts-ignore */}
          <div className="relative glass-card rounded-[2rem] overflow-hidden shadow-2xl h-[50vh] lg:h-[600px] bg-black/40 flex items-center justify-center">
            {result.result_image ? (
              // @ts-ignore
              <div className="relative w-full h-full flex items-center justify-center p-4">
                <img 
                  src={`data:image/jpeg;base64,${result.result_image}`} 
                  alt="Analyzed Face" 
                  className="w-full h-full object-contain rounded-xl" 
                />
                {/* Tech Overlay - Scanning Line */}
                {/* @ts-ignore */}
                <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-pastel/20 to-transparent animate-scan opacity-60 h-32 w-full border-t border-b border-pastel/40 box-content"></div>
              </div>
            ) : (
              // @ts-ignore
              <div className="w-full h-full flex items-center justify-center">
                {/* @ts-ignore */}
                <span className="text-gray-400">No image available</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Stats & Data */}
        {/* @ts-ignore */}
        <div className="lg:col-span-5 space-y-6">
          
          {/* Metrics Grid */}
          {/* @ts-ignore */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Acne', value: result.acne_count, sev: result.acne_severity },
              { label: 'Blemishes', value: result.blemish_count, sev: result.blemish_severity },
              { label: 'Oiliness', value: Math.round(result.oiliness_score) + '%', sev: result.oiliness_severity }
            ].map((item, i) => (
              // @ts-ignore
              <div key={item.label} className="glass-card p-4 rounded-2xl flex flex-col items-center justify-center hover:bg-white/10 transition-all duration-300 border-white/5">
                {/* @ts-ignore */}
                <span className="text-3xl font-light text-white text-glow">{item.value}</span>
                {/* @ts-ignore */}
                <span className="text-[10px] uppercase tracking-widest text-gray-400 mt-2 mb-1">{item.label}</span>
                {/* @ts-ignore */}
                <span className={`text-[10px] font-bold uppercase ${getSeverityColor(item.sev)}`}>{item.sev}</span>
              </div>
            ))}
          </div>

          {/* Formula Card */}
          {/* @ts-ignore */}
          <div className="glass-card rounded-[2rem] p-8 relative overflow-hidden border-white/10">
            {/* @ts-ignore */}
            <div className="absolute top-0 right-0 w-40 h-40 bg-pastel rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none opacity-20"></div>
            
            {/* @ts-ignore */}
            <div className="flex justify-between items-end mb-8 relative z-10">
              {/* @ts-ignore */}
              <div>
                {/* @ts-ignore */}
                <h3 className="text-xl font-semibold text-white tracking-tight text-glow">Your Formula</h3>
                {/* @ts-ignore */}
                <p className="text-sm text-gray-400 mt-1">Personalized daily regimen</p>
              </div>
              {/* @ts-ignore */}
              <div className="text-right">
                {/* @ts-ignore */}
                <span className="text-3xl font-light text-pastel text-glow">{result.dispenser.total_ml}</span>
                {/* @ts-ignore */}
                <span className="text-sm text-gray-400 ml-1">ml total</span>
              </div>
            </div>

            {/* Bar Chart */}
            {/* @ts-ignore */}
            <div className="h-4 flex rounded-full overflow-hidden bg-black/40 mb-8 border border-white/5">
              {/* @ts-ignore */}
              <div style={{ width: `${result.dispenser.cleanser_pct}%` }} className="h-full bg-pastel transition-all duration-1000 ease-out relative group shadow-[0_0_10px_rgba(188,226,191,0.5)]"></div>
              {/* @ts-ignore */}
              <div style={{ width: `${result.dispenser.treatment_pct}%` }} className="h-full bg-emerald-500 transition-all duration-1000 ease-out delay-100 relative group"></div>
              {/* @ts-ignore */}
              <div style={{ width: `${result.dispenser.moisturizer_pct}%` }} className="h-full bg-gray-500 transition-all duration-1000 ease-out delay-200 relative group"></div>
            </div>

            {/* Legend */}
            {/* @ts-ignore */}
            <div className="space-y-4">
              {/* @ts-ignore */}
              <div className="flex items-center justify-between group">
                {/* @ts-ignore */}
                <div className="flex items-center gap-3">
                  {/* @ts-ignore */}
                  <div className="w-3 h-3 rounded-full bg-pastel shadow-[0_0_8px_rgba(188,226,191,0.6)]" />
                  {/* @ts-ignore */}
                  <span className="text-sm text-gray-300 font-medium">Cleanser</span>
                </div>
                {/* @ts-ignore */}
                <span className="text-sm font-bold text-white">{result.dispenser.cleanser_ml}ml</span>
              </div>
              {/* @ts-ignore */}
              <div className="flex items-center justify-between group">
                {/* @ts-ignore */}
                <div className="flex items-center gap-3">
                  {/* @ts-ignore */}
                  <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  {/* @ts-ignore */}
                  <span className="text-sm text-gray-300 font-medium">Treatment</span>
                </div>
                {/* @ts-ignore */}
                <span className="text-sm font-bold text-white">{result.dispenser.treatment_ml}ml</span>
              </div>
              {/* @ts-ignore */}
              <div className="flex items-center justify-between group">
                {/* @ts-ignore */}
                <div className="flex items-center gap-3">
                  {/* @ts-ignore */}
                  <div className="w-3 h-3 rounded-full bg-gray-500" />
                  {/* @ts-ignore */}
                  <span className="text-sm text-gray-300 font-medium">Moisturizer</span>
                </div>
                {/* @ts-ignore */}
                <span className="text-sm font-bold text-white">{result.dispenser.moisturizer_ml}ml</span>
              </div>
            </div>
          </div>

          {/* Region Breakdown */}
          {/* @ts-ignore */}
          <div className="glass-card rounded-2xl p-6 border-white/10">
            {/* @ts-ignore */}
            <h3 className="text-xs uppercase tracking-[0.2em] font-bold text-gray-400 mb-6">Regional Analysis</h3>
            {/* @ts-ignore */}
            <div className="space-y-5">
              {Object.entries(result.regions).map(([region, count], index) => (
                // @ts-ignore
                <div key={region} className="flex items-center gap-4">
                  {/* @ts-ignore */}
                  <span className="w-24 text-xs font-medium text-gray-400 capitalize">{region.replace(/_/g, ' ')}</span>
                  {/* @ts-ignore */}
                  <div className="flex-1 h-1.5 bg-black/40 rounded-full overflow-hidden border border-white/5">
                    {/* @ts-ignore */}
                    <div 
                      className="h-full bg-pastel rounded-full shadow-[0_0_8px_rgba(188,226,191,0.5)]" 
                      style={{ 
                        width: `${Math.min(100, (count as number) * 20)}%`,
                        animation: `slideRight 1s ease-out ${index * 0.1}s backwards`
                      }} 
                    />
                  </div>
                  {/* @ts-ignore */}
                  <span className="w-8 text-right text-xs font-bold text-white">{count}</span>
                </div>
              ))}
            </div>
          </div>

          {/* @ts-ignore */}
          <div className="pt-4 space-y-3">
            <Button onClick={handlePrintFormula} variant="primary" className="w-full py-4 text-sm tracking-widest uppercase" disabled={isPrinting}>
              {isPrinting ? 'Printing...' : 'Print Formula'}
            </Button>
            <Button onClick={onReset} variant="outline" className="w-full py-4 text-sm tracking-widest uppercase">
              Start New Analysis
            </Button>
          </div>

        </div>
      </div>
      
      {/* @ts-ignore */}
      <style>{`
        @keyframes slideRight {
          from { width: 0; }
        }
      `}</style>
    </div>
  );
};

// --- Main App Implementation ---
// Replace this with your actual API endpoint if different
const API_URL = 'http://localhost:8000';
const ESP32_IP = '172.20.10.6'; 

export default function HomeScreen() {
  const [viewState, setViewState] = useState<ViewState>('landing');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize camera stream
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } 
      });
      setCameraStream(stream);
      setViewState('camera');
    } catch (err) {
      console.error(err);
      setError("Unable to access camera. Please verify permissions.");
      setViewState('error');
    }
  };

  // Effect to attach stream to video element when viewState becomes 'camera'
  useEffect(() => {
    if (viewState === 'camera' && cameraStream && videoRef.current) {
        videoRef.current.srcObject = cameraStream;
        videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(e => console.error("Auto-play blocked:", e));
        };
    }
  }, [viewState, cameraStream]);

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        stopCamera();
        analyzeImage(base64);
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = (reader.result as string).split(',')[1];
        analyzeImage(base64String);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async (base64Image: string) => {
    setViewState('analyzing');
    setError(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

      const response = await fetch(`${API_URL}/analyze-base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ image: base64Image }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error('Analysis service returned an error.');
      }

      const data: AnalysisResult = await response.json();
      setAnalysisResult(data);
      setViewState('results');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to connect to analysis server');
      setViewState('error');
    }
  };

  const reset = () => {
    stopCamera();
    setAnalysisResult(null);
    setError(null);
    setViewState('landing');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // If not on web, show warning
  if (Platform.OS !== 'web') {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#1a2e1f' }}>
        <Text style={{ color: '#fff' }}>This app is optimized for Web.</Text>
      </View>
    );
  }

  return (
    <>
    <Stack.Screen options={{ headerShown: false }} />
    <WebStyleInjector />
    {/* @ts-ignore */}
    <div className="min-h-screen relative selection:bg-pastel selection:text-forest-bg pb-10 flex flex-col">
      
      {/* Background Visual Effects */}
      {/* @ts-ignore */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        {/* Glows */}
        {/* @ts-ignore */}
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-pastel/10 blur-[100px] animate-pulse-slow"></div>
        {/* @ts-ignore */}
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-pastel/5 blur-[100px] animate-pulse-slow" style={{ animationDelay: '1.5s' }}></div>
        
        {/* Floating Icons/Elements */}
        {/* @ts-ignore */}
        <div className="absolute top-[20%] left-[10%] text-pastel/20 text-4xl animate-float" style={{ animationDelay: '0s' }}>✦</div>
        {/* @ts-ignore */}
        <div className="absolute bottom-[20%] right-[10%] text-pastel/10 text-6xl animate-float" style={{ animationDelay: '2s' }}>✧</div>
      </div>

      {/* Header */}
      {viewState !== 'landing' && (
        // @ts-ignore
        <header className="relative z-50 pt-8 px-8 flex justify-end">
          {/* @ts-ignore */}
          <button onClick={reset} className="text-gray-400 hover:text-white transition-colors text-xs font-bold uppercase tracking-widest border border-transparent hover:border-gray-600 px-4 py-2 rounded-lg">
            Exit
          </button>
        </header>
      )}

      {/* @ts-ignore */}
      <main className="relative z-10 container mx-auto px-6 py-12 flex flex-col items-center justify-center flex-grow">
        
        {/* LANDING STATE */}
        {viewState === 'landing' && (
          // @ts-ignore
          <div className="flex flex-col items-center text-center max-w-4xl animate-fade-in-up mt-10">
            {/* Branding - Tiny Text Logo */}
            {/* @ts-ignore */}
            <div className="mb-4">
               {/* @ts-ignore */}
               <span className="text-xs md:text-sm font-bold tracking-[0.4em] uppercase text-pastel text-glow">
                 glowstate
               </span>
            </div>

            {/* Main Headline - UPDATED to White with Glow */}
            {/* @ts-ignore */}
            <h1 className="relative text-5xl md:text-7xl font-semibold tracking-tighter text-white mb-6 leading-tight text-glow-strong">
              Reveal your true<br/>skin profile.
            </h1>
            
            {/* @ts-ignore */}
            <p className="text-lg text-gray-300 mb-12 max-w-lg leading-relaxed font-light">
              Skin analysis powered by artificial intelligence. 
              Get personalized solutions in seconds.
            </p>

            {/* @ts-ignore */}
            <div className="flex flex-col sm:flex-row gap-6 w-full sm:w-auto">
              <Button onClick={startCamera} variant="primary">
                Start Scan
              </Button>
              <Button onClick={() => fileInputRef.current?.click()} variant="secondary">
                Upload Photo
              </Button>
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept="image/*"
                onChange={handleFileUpload}
              />
            </div>

            {/* @ts-ignore */}
            <div className="mt-24 grid grid-cols-3 gap-16 text-center border-t border-white/10 pt-12">
              {[
                { label: 'Detection', val: '99%' },
                { label: 'Analysis', val: '< 2s' },
                { label: 'Privacy', val: '100%' },
              ].map((stat) => (
                // @ts-ignore
                <div key={stat.label} className="flex flex-col group cursor-default">
                  {/* @ts-ignore */}
                  <span className="text-3xl md:text-4xl font-light text-white mb-2 group-hover:text-pastel group-hover:text-glow transition-all">{stat.val}</span>
                  {/* @ts-ignore */}
                  <span className="text-[10px] uppercase tracking-widest text-gray-500 group-hover:text-white transition-colors">{stat.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* CAMERA STATE - Fixed Ref Logic */}
        {viewState === 'camera' && (
          // @ts-ignore
          <div className="relative w-full max-w-4xl bg-black rounded-[2rem] overflow-hidden shadow-[0_0_50px_rgba(188,226,191,0.1)] border border-white/10 animate-fade-in flex flex-col items-center justify-center">
             
            {/* Container for Video - Explicit Height for Mobile/Desktop */}
            {/* @ts-ignore */}
            <div className="relative w-full h-[60vh] md:h-[70vh] flex items-center justify-center bg-black">
                {/* Video Element - Simplified Positioning */}
                {/* @ts-ignore */}
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted
                  className="w-full h-full object-cover transform scale-x-[-1]"
                />
            </div>
            
            <canvas ref={canvasRef} className="hidden" />
            
            {/* Camera Overlay */}
            {/* @ts-ignore */}
            <div className="absolute inset-0 z-20 flex flex-col justify-between pointer-events-none">
               {/* Gradients */}
              {/* @ts-ignore */}
              <div className="w-full h-32 bg-gradient-to-b from-black/50 to-transparent"></div>
              {/* @ts-ignore */}
              <div className="w-full h-40 bg-gradient-to-t from-black/50 to-transparent"></div>
              
              {/* Scanning Frame */}
              {/* @ts-ignore */}
              <div className="absolute inset-8 border border-white/20 rounded-2xl">
                {/* @ts-ignore */}
                <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-pastel rounded-tl-lg shadow-[0_0_15px_rgba(188,226,191,0.5)]"></div>
                {/* @ts-ignore */}
                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-pastel rounded-tr-lg shadow-[0_0_15px_rgba(188,226,191,0.5)]"></div>
                {/* @ts-ignore */}
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-pastel rounded-bl-lg shadow-[0_0_15px_rgba(188,226,191,0.5)]"></div>
                {/* @ts-ignore */}
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-pastel rounded-br-lg shadow-[0_0_15px_rgba(188,226,191,0.5)]"></div>
              </div>
              
              {/* Capture Button */}
              {/* @ts-ignore */}
              <div className="absolute bottom-10 inset-x-0 flex items-center justify-center pointer-events-auto">
                {/* @ts-ignore */}
                <button 
                  onClick={captureImage}
                  className="w-20 h-20 rounded-full border-4 border-white/30 flex items-center justify-center hover:border-pastel hover:scale-105 transition-all duration-300 group bg-white/10 backdrop-blur-sm"
                >
                  {/* @ts-ignore */}
                  <div className="w-16 h-16 bg-white rounded-full group-hover:scale-90 transition-transform duration-300 shadow-[0_0_20px_rgba(255,255,255,0.5)]"></div>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ANALYZING STATE */}
        {viewState === 'analyzing' && (
          // @ts-ignore
          <div className="flex flex-col items-center justify-center animate-fade-in">
            {/* @ts-ignore */}
            <div className="relative w-32 h-32 mb-8">
              {/* @ts-ignore */}
              <div className="absolute inset-0 border-4 border-white/10 rounded-full"></div>
              {/* @ts-ignore */}
              <div className="absolute inset-0 border-4 border-pastel rounded-full border-t-transparent animate-spin"></div>
              {/* @ts-ignore */}
              <div className="absolute inset-4 bg-white/5 rounded-full flex items-center justify-center shadow-inner backdrop-blur-sm">
                {/* @ts-ignore */}
                <div className="w-2 h-2 bg-pastel rounded-full animate-pulse shadow-[0_0_10px_#BCE2BF]"></div>
              </div>
            </div>
            {/* @ts-ignore */}
            <h2 className="text-2xl font-light text-white tracking-wide mb-2 animate-pulse text-glow">Processing Scan</h2>
            {/* @ts-ignore */}
            <p className="text-sm text-gray-500 tracking-widest uppercase">Identifying dermatological features</p>
          </div>
        )}

        {/* RESULTS STATE */}
        {viewState === 'results' && analysisResult && (
          <AnalysisResultView result={analysisResult} onReset={reset} />
        )}

        {/* ERROR STATE */}
        {viewState === 'error' && (
          // @ts-ignore
          <div className="glass-card p-12 rounded-[2rem] max-w-md text-center animate-fade-in border border-rose-500/20">
            {/* @ts-ignore */}
            <div className="w-16 h-16 bg-rose-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
              {/* @ts-ignore */}
              <span className="text-2xl text-rose-400">!</span>
            </div>
            {/* @ts-ignore */}
            <h3 className="text-xl text-white font-medium mb-3">Analysis Failed</h3>
            {/* @ts-ignore */}
            <p className="text-gray-400 mb-8 leading-relaxed">{error || "Something went wrong during the analysis."}</p>
            <Button onClick={reset} variant="secondary">Try Again</Button>
          </div>
        )}

      </main>
    </div>
    </>
  );
}
