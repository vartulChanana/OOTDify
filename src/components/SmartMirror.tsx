import React, { useRef, useEffect, useState } from 'react';
import { Camera, Star, Sparkles, Shirt, Palette, TrendingUp, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { pipeline, env } from '@huggingface/transformers';

// Configure transformers
env.allowLocalModels = false;
env.useBrowserCache = true;

interface OutfitAnalysis {
  score: number;
  style: string;
  colors: string[];
  suggestions: string[];
  confidence: number;
}

interface PersonDetection {
  bbox: [number, number, number, number]; // [x, y, width, height]
  confidence: number;
}

const SmartMirror = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [analysis, setAnalysis] = useState<OutfitAnalysis | null>(null);
  const [personDetected, setPersonDetected] = useState<PersonDetection | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [detector, setDetector] = useState<any>(null);
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    startCamera();
    loadDetectionModel();
  }, []);

  // Load the person detection model
  const loadDetectionModel = async () => {
    try {
      console.log('Loading person detection model...');
      const objectDetector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
        device: 'webgpu',
      });
      setDetector(objectDetector);
      console.log('Person detection model loaded successfully');
    } catch (error) {
      console.error('Error loading detection model:', error);
      // Fallback without webgpu
      try {
        const objectDetector = await pipeline('object-detection', 'Xenova/detr-resnet-50');
        setDetector(objectDetector);
        console.log('Person detection model loaded successfully (fallback)');
      } catch (fallbackError) {
        console.error('Error loading detection model (fallback):', fallbackError);
      }
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  // Detect person in the video feed
  const detectPerson = async () => {
    if (!videoRef.current || !detector) return null;

    try {
      // Create canvas to capture frame
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0);

      // Convert to base64 for the model
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      // Detect objects
      const detections = await detector(imageData);
      
      // Find person with highest confidence
      const personDetection = detections
        .filter((det: any) => det.label === 'person')
        .reduce((best: any, current: any) => 
          current.score > (best?.score || 0) ? current : best, null);

      if (personDetection) {
        const bbox: [number, number, number, number] = [
          personDetection.box.xmin,
          personDetection.box.ymin,
          personDetection.box.xmax - personDetection.box.xmin,
          personDetection.box.ymax - personDetection.box.ymin
        ];
        
        return {
          bbox,
          confidence: personDetection.score
        };
      }
    } catch (error) {
      console.error('Error detecting person:', error);
    }
    return null;
  };

  // Capture and analyze the image
  const analyzeOutfit = async () => {
    if (!videoRef.current) return;
    
    setIsAnalyzing(true);
    
    try {
      // First detect person
      console.log('Detecting person...');
      const detection = await detectPerson();
      setPersonDetected(detection);

      // Capture the current frame
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0);
      
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
      setCapturedImage(imageDataUrl);

      // Analyze image for realistic scoring
      const analysisCanvas = document.createElement('canvas');
      const analysisCtx = analysisCanvas.getContext('2d');
      if (!analysisCtx) return;

      analysisCanvas.width = canvas.width;
      analysisCanvas.height = canvas.height;
      analysisCtx.drawImage(videoRef.current, 0, 0);
      
      const imageData = analysisCtx.getImageData(0, 0, analysisCanvas.width, analysisCanvas.height);
      const colorAnalysis = analyzeImageColors(imageData);
      
      await new Promise(resolve => setTimeout(resolve, 2000));

      const newAnalysis: OutfitAnalysis = {
        score: colorAnalysis.score,
        style: colorAnalysis.style,
        colors: colorAnalysis.dominantColors,
        suggestions: colorAnalysis.suggestions,
        confidence: colorAnalysis.confidence
      };

      setAnalysis(newAnalysis);
      setShowResults(true);
    } catch (error) {
      console.error('Error analyzing outfit:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Draw person highlight overlay
  useEffect(() => {
    if (!canvasRef.current || !videoRef.current || !personDetected) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = videoRef.current.clientWidth;
    canvas.height = videoRef.current.clientHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale factors
    const scaleX = canvas.width / videoRef.current.videoWidth;
    const scaleY = canvas.height / videoRef.current.videoHeight;

    // Draw person highlight
    const [x, y, width, height] = personDetected.bbox;
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;
    const scaledWidth = width * scaleX;
    const scaledHeight = height * scaleY;

    // Draw highlight rectangle
    ctx.strokeStyle = 'hsl(var(--primary))';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

    // Draw confidence badge
    ctx.fillStyle = 'hsl(var(--primary))';
    ctx.fillRect(scaledX, scaledY - 30, 120, 25);
    ctx.fillStyle = 'white';
    ctx.font = '12px sans-serif';
    ctx.fillText(`Person: ${Math.round(personDetected.confidence * 100)}%`, scaledX + 5, scaledY - 10);
  }, [personDetected]);

  const getScoreColor = (score: number) => {
    if (score >= 9) return 'text-green-400';
    if (score >= 7) return 'text-blue-400';
    if (score >= 5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const analyzeImageColors = (imageData: ImageData) => {
    const data = imageData.data;
    const colorCounts: { [key: string]: number } = {};
    let totalBrightness = 0;
    let contrastScore = 0;
    let colorVariety = 0;

    // Sample pixels (every 50th pixel for performance)
    for (let i = 0; i < data.length; i += 200) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      const brightness = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
      totalBrightness += brightness;
      
      // Categorize colors
      const hue = rgbToHue(r, g, b);
      const colorCategory = getColorCategory(hue, brightness);
      colorCounts[colorCategory] = (colorCounts[colorCategory] || 0) + 1;
    }

    const avgBrightness = totalBrightness / (data.length / 4);
    const dominantColors = Object.keys(colorCounts)
      .sort((a, b) => colorCounts[b] - colorCounts[a])
      .slice(0, 3);

    colorVariety = Object.keys(colorCounts).length;
    contrastScore = calculateContrast(colorCounts, avgBrightness);

    // Calculate score based on harmony, contrast, and brightness
    let score = 5; // Base score
    
    // Brightness optimization (0.3-0.7 is ideal)
    if (avgBrightness >= 0.3 && avgBrightness <= 0.7) score += 2;
    else if (avgBrightness >= 0.2 && avgBrightness <= 0.8) score += 1;
    
    // Color harmony (2-4 colors is ideal)
    if (colorVariety >= 2 && colorVariety <= 4) score += 2;
    else if (colorVariety >= 1 && colorVariety <= 5) score += 1;
    
    // Contrast score
    score += Math.min(contrastScore, 2);
    
    // Style determination
    const style = determineStyle(dominantColors, avgBrightness);
    
    // Generate suggestions
    const suggestions = generateSuggestions(dominantColors, avgBrightness, contrastScore);

    return {
      score: Math.min(Math.max(Math.round(score), 1), 10),
      style,
      dominantColors,
      suggestions,
      confidence: Math.round(85 + (score - 5) * 3)
    };
  };

  const rgbToHue = (r: number, g: number, b: number) => {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const diff = max - min;
    let hue = 0;
    
    if (diff !== 0) {
      switch (max) {
        case r: hue = ((g - b) / diff) % 6; break;
        case g: hue = (b - r) / diff + 2; break;
        case b: hue = (r - g) / diff + 4; break;
      }
    }
    return Math.round(hue * 60);
  };

  const getColorCategory = (hue: number, brightness: number) => {
    if (brightness < 0.2) return 'Black';
    if (brightness > 0.8) return 'White';
    
    if (hue >= 0 && hue < 30) return 'Red';
    if (hue >= 30 && hue < 60) return 'Orange';
    if (hue >= 60 && hue < 120) return 'Green';
    if (hue >= 120 && hue < 180) return 'Cyan';
    if (hue >= 180 && hue < 240) return 'Blue';
    if (hue >= 240 && hue < 300) return 'Purple';
    return 'Pink';
  };

  const calculateContrast = (colorCounts: { [key: string]: number }, avgBrightness: number) => {
    const hasLight = avgBrightness > 0.6;
    const hasDark = avgBrightness < 0.4;
    return hasLight && hasDark ? 2 : hasLight || hasDark ? 1 : 0;
  };

  const determineStyle = (colors: string[], brightness: number) => {
    const colorSet = new Set(colors);
    
    if (colorSet.has('Black') && colorSet.has('White')) return 'Professional';
    if (colorSet.has('Blue') && (colorSet.has('White') || colorSet.has('Black'))) return 'Business Casual';
    if (brightness > 0.7) return 'Fresh & Light';
    if (colors.includes('Red') || colors.includes('Orange')) return 'Bold & Confident';
    if (colorSet.has('Green')) return 'Natural & Relaxed';
    if (brightness < 0.3) return 'Elegant & Sophisticated';
    return 'Casual';
  };

  const generateSuggestions = (colors: string[], brightness: number, contrast: number) => {
    const suggestions = [];
    
    if (brightness < 0.3) suggestions.push('Try adding a lighter accent piece for better balance');
    if (brightness > 0.8) suggestions.push('Consider a darker accessory for contrast');
    if (contrast < 1) suggestions.push('Add more contrast with different tones');
    if (colors.length < 2) suggestions.push('Introduce a complementary color');
    if (colors.length > 4) suggestions.push('Simplify with fewer colors for cleaner look');
    
    const positives = [
      'Great color coordination!',
      'Nice style choice!',
      'Well-balanced outfit!'
    ];
    
    if (suggestions.length === 0) {
      suggestions.push(positives[Math.floor(Math.random() * positives.length)]);
    }
    
    return suggestions.slice(0, 2);
  };

  const renderStars = (score: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        className={`w-4 h-4 ${
          i < Math.floor(score / 2) 
            ? 'fill-primary text-primary' 
            : 'text-muted-foreground'
        }`}
      />
    ));
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background glow effects */}
      <div className="absolute inset-0 bg-gradient-glow opacity-30"></div>
      
      {/* Mirror frame */}
      <div className="relative h-screen p-8">
        <div className="h-full bg-gradient-glass backdrop-blur-sm rounded-3xl border border-border shadow-mirror relative overflow-hidden">
          
          {/* Camera feed */}
          <div className="absolute inset-4 rounded-2xl overflow-hidden bg-mirror-surface relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover scale-x-[-1]"
            />
            
            {/* Person detection overlay canvas */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full scale-x-[-1] pointer-events-none"
            />
            
            {/* Overlay UI */}
            <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent">
              
              {/* Header */}
              <div className="absolute top-6 left-6 right-6 flex justify-between items-start">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-primary rounded-full animate-pulse shadow-glow"></div>
                  <span className="text-sm font-medium text-foreground/80">
                    Smart Mirror Active {personDetected && `- Person Detected`}
                  </span>
                </div>
                
                {cameraActive && (
                  <Badge variant="secondary" className="bg-primary/20 text-primary border-primary/30">
                    <Camera className="w-3 h-3 mr-1" />
                    Live
                  </Badge>
                )}
              </div>

              {/* Captured Image Preview */}
              {capturedImage && (
                <div className="absolute top-20 right-6 w-32 h-24 rounded-lg overflow-hidden border-2 border-primary/50">
                  <img 
                    src={capturedImage} 
                    alt="Captured for analysis" 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute bottom-0 left-0 right-0 bg-primary/80 text-white text-xs text-center py-1">
                    Analyzed
                  </div>
                </div>
              )}

              {/* Analysis Results - only show when modal is closed */}
              {analysis && !showResults && (
                <div className="absolute bottom-6 left-6 right-6 space-y-4">
                  
                  {/* Main Score Card */}
                  <Card className="bg-gradient-glass backdrop-blur-md border-border/50 shadow-accent">
                    <div className="p-6">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <Sparkles className="w-6 h-6 text-primary" />
                          <span className="text-lg font-semibold">Outfit Analysis</span>
                        </div>
                        <div className="text-right">
                          <div className={`text-3xl font-bold ${getScoreColor(analysis.score)}`}>
                            {analysis.score}/10
                          </div>
                          <div className="flex gap-1 mt-1">
                            {renderStars(analysis.score)}
                          </div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="flex items-center gap-2">
                          <Shirt className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm">Style: {analysis.style}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm">Confidence: {analysis.confidence}%</span>
                        </div>
                      </div>
                    </div>
                  </Card>

                  {/* Color Analysis */}
                  <Card className="bg-gradient-glass backdrop-blur-md border-border/50">
                    <div className="p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <Palette className="w-4 h-4 text-accent" />
                        <span className="text-sm font-medium">Color Palette</span>
                      </div>
                      <div className="flex gap-2">
                        {analysis.colors.map((color, index) => (
                          <Badge key={index} variant="secondary" className="text-xs">
                            {color}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </Card>

                  {/* Suggestions */}
                  {analysis.suggestions.length > 0 && (
                    <Card className="bg-gradient-glass backdrop-blur-md border-border/50">
                      <div className="p-4">
                        <h3 className="text-sm font-medium mb-2 text-accent">Style Suggestions</h3>
                        <div className="space-y-1">
                          {analysis.suggestions.map((suggestion, index) => (
                            <p key={index} className="text-xs text-muted-foreground">
                              • {suggestion}
                            </p>
                          ))}
                        </div>
                      </div>
                    </Card>
                  )}
                </div>
              )}

              {/* Analyzing indicator */}
              {isAnalyzing && (
                <div className="absolute bottom-6 left-6 right-6">
                  <Card className="bg-gradient-glass backdrop-blur-md border-border/50">
                    <div className="p-6 text-center">
                      <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-3"></div>
                      <p className="text-sm text-muted-foreground">
                        {!detector ? 'Loading AI model...' : 'Detecting person and analyzing outfit...'}
                      </p>
                    </div>
                  </Card>
                </div>
              )}

              {/* Manual analysis button */}
              <div className="absolute bottom-6 right-6">
                <Button
                  onClick={analyzeOutfit}
                  disabled={isAnalyzing || !detector}
                  className="bg-gradient-primary hover:shadow-glow transition-all duration-300"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  {!detector ? 'Loading...' : 'Analyze Now'}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results Modal */}
      <Dialog open={showResults} onOpenChange={(open) => {
        if (!open) {
          // Clear all analysis data when modal closes
          setShowResults(false);
          setAnalysis(null);
          setCapturedImage(null);
          setPersonDetected(null);
        } else {
          setShowResults(open);
        }
      }}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-gradient-glass backdrop-blur-md border-border/50">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <Sparkles className="w-6 h-6 text-primary" />
              Outfit Analysis Results
            </DialogTitle>
          </DialogHeader>
          
          {capturedImage && analysis && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
              {/* Analyzed Image */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Analyzed Photo</h3>
                <div className="relative rounded-xl overflow-hidden border-2 border-primary/30">
                  <img 
                    src={capturedImage} 
                    alt="Analyzed outfit" 
                    className="w-full h-auto object-cover"
                  />
                  {personDetected && (
                    <div className="absolute top-2 right-2 bg-primary/80 text-white px-2 py-1 rounded text-xs">
                      Person Detected: {Math.round(personDetected.confidence * 100)}%
                    </div>
                  )}
                </div>
              </div>

              {/* Analysis Results */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Analysis Details</h3>
                
                {/* Score */}
                <Card className="bg-gradient-glass backdrop-blur-md border-border/50">
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-medium">Overall Score</span>
                      <div className="text-right">
                        <div className={`text-2xl font-bold ${getScoreColor(analysis.score)}`}>
                          {analysis.score}/10
                        </div>
                        <div className="flex gap-1 mt-1 justify-end">
                          {renderStars(analysis.score)}
                        </div>
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Style & Confidence */}
                <Card className="bg-gradient-glass backdrop-blur-md border-border/50">
                  <div className="p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Shirt className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm font-medium">Style Category</span>
                      </div>
                      <Badge variant="secondary">{analysis.style}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm font-medium">AI Confidence</span>
                      </div>
                      <span className="text-sm font-semibold">{analysis.confidence}%</span>
                    </div>
                  </div>
                </Card>

                {/* Color Palette */}
                <Card className="bg-gradient-glass backdrop-blur-md border-border/50">
                  <div className="p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Palette className="w-4 h-4 text-accent" />
                      <span className="text-sm font-medium">Detected Colors</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {analysis.colors.map((color, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {color}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </Card>

                {/* Suggestions */}
                {analysis.suggestions.length > 0 && (
                  <Card className="bg-gradient-glass backdrop-blur-md border-border/50">
                    <div className="p-4">
                      <h4 className="text-sm font-medium mb-3 text-accent">Style Suggestions</h4>
                      <div className="space-y-2">
                        {analysis.suggestions.map((suggestion, index) => (
                          <p key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                            <span className="text-primary mt-1">•</span>
                            {suggestion}
                          </p>
                        ))}
                      </div>
                    </div>
                  </Card>
                )}

                {/* Action Buttons */}
                <div className="flex gap-2 pt-2">
                  <Button 
                    onClick={() => setShowResults(false)}
                    variant="outline"
                    className="flex-1"
                  >
                    Close Results
                  </Button>
                  <Button 
                    onClick={() => {
                      setShowResults(false);
                      setAnalysis(null);
                      setCapturedImage(null);
                      setPersonDetected(null);
                    }}
                    className="flex-1 bg-gradient-primary"
                  >
                    Analyze Again
                  </Button>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default SmartMirror;
