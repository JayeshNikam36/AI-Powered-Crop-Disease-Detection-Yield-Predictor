// src/dashboard/src/pages/Detection.tsx
import { useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export default function Detection() {
  const [uploading, setUploading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null); // data URL for preview
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);

  const API_BASE = import.meta.env.VITE_API_URL ?? ""; // set VITE_API_URL if needed

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file");
      return;
    }
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result as string);
    };
    reader.readAsDataURL(file);
    // reset previous result
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast.error("Please select an image first");
      return;
    }
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", selectedFile);

      const res = await fetch(`/api/predict`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const txt = await res.text();
        toast.error("Analysis failed: " + (txt || res.statusText));
        setUploading(false);
        return;
      }

      const data = await res.json();
      setResult(data);
      toast.success("Analysis complete");
    } catch (err) {
      console.error(err);
      toast.error("Network error: could not call API");
    } finally {
      setUploading(false);
    }
  };

  // helper to convert base64 from backend into a data url for <img>
  const base64ToDataUrl = (b64?: string) => (b64 ? `data:image/png;base64,${b64}` : undefined);

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Disease Detection</h1>
        <p className="text-muted-foreground">Upload a crop leaf image for AI-powered disease analysis</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle>Upload Image</CardTitle>
            <CardDescription>Select a clear photo of the crop leaf</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="border-2 border-dashed border-border rounded-lg p-8 hover:border-primary transition-colors">
              <label className="flex flex-col items-center justify-center gap-4 cursor-pointer">
                <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                  <Upload className="h-8 w-8 text-primary" />
                </div>
                <div className="text-center">
                  <p className="font-medium">Click to upload or drag and drop</p>
                  <p className="text-sm text-muted-foreground mt-1">PNG, JPG, JPEG up to 10MB</p>
                </div>
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              </label>
            </div>

            {selectedImage && (
              <div className="space-y-4">
                <img src={selectedImage} alt="Selected crop" className="w-full rounded-lg border shadow-md" />
                <Button onClick={handleAnalyze} disabled={uploading} className="w-full" size="lg">
                  {uploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze Image"
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle>Detection Results</CardTitle>
            <CardDescription>AI analysis of the uploaded image</CardDescription>
          </CardHeader>
          <CardContent>
            {!selectedImage ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="h-20 w-20 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Upload className="h-10 w-10 text-muted-foreground" />
                </div>
                <p className="text-muted-foreground">Upload an image to see results</p>
              </div>
            ) : result ? (
              <div className="space-y-6">
                {/* Predicted */}
                <div className="space-y-2">
                  <h3 className="font-semibold text-lg">Predicted Disease</h3>
                  <div className="p-4 bg-gradient-card rounded-lg border">
                    <p className="text-2xl font-bold text-primary">{result.predicted_class}</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Confidence: {(result.score * 100).toFixed(1)}%
                    </p>
                    <p className="mt-2 text-sm">{result.description || "No description available."}</p>
                  </div>
                </div>

                {/* Images */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <h4 className="text-sm font-medium mb-2">Original</h4>
                    <img
                      src={result.original_image_base64 ? base64ToDataUrl(result.original_image_base64) : selectedImage}
                      alt="original"
                      className="w-full rounded-md border"
                    />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-2">Grad-CAM Overlay</h4>
                    <img
                      src={base64ToDataUrl(result.gradcam_overlay_base64)}
                      alt="overlay"
                      className="w-full rounded-md border"
                    />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-2">Heatmap</h4>
                    <img
                      src={base64ToDataUrl(result.gradcam_heatmap_base64)}
                      alt="heatmap"
                      className="w-full rounded-md border"
                    />
                  </div>
                </div>

                {/* Top Predictions */}
                <div>
                  <h3 className="font-semibold">Top Predictions</h3>
                  <div className="space-y-2 mt-2">
                    {(result.probabilities_by_class ?? [])
                      .slice()
                      .sort((a: any, b: any) => b.probability - a.probability)
                      .slice(0, 5)
                      .map((p: any) => (
                        <div key={p.class} className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">{p.class}</span>
                          <span className="font-medium">{(p.probability * 100).toFixed(2)}%</span>
                        </div>
                      ))}
                  </div>
                </div>

                <Button variant="outline" className="w-full" asChild>
                  <a href="/knowledge">View Disease Information</a>
                </Button>
              </div>
            ) : (
              <div className="flex items-center justify-center py-12 text-center">
                <div className="text-muted-foreground">Click Analyze to call the model and see results</div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
