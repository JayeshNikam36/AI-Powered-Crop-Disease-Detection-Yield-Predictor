// src/dashboard/src/pages/GradCAM.tsx
import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { Brain, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function GradCAM() {
  const location = useLocation();
  const [data, setData] = useState<any | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [opacity, setOpacity] = useState<number>(0.5);

  useEffect(() => {
    const fromState = (location.state as any) ?? null;
    if (fromState) {
      setData(fromState);
    } else {
      const sess = sessionStorage.getItem("last_result");
      if (sess) setData(JSON.parse(sess));
      const prev = sessionStorage.getItem("last_image_preview");
      if (prev) setPreview(prev);
    }
  }, [location.state]);

  // If the backend didn't return the original preview, we may only have the overlay image
  useEffect(() => {
    if (!preview && data?.gradcam_overlay_base64) {
      // sometimes overlay contains original + heatmap; we still prefer original preview if present
      // nothing to set here if backend did not return original
    }
  }, [data, preview]);

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Grad-CAM Analysis</h1>
        <p className="text-muted-foreground">Visual explanations of model predictions</p>
      </div>

      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>What is Grad-CAM?</AlertTitle>
        <AlertDescription>
          Grad-CAM highlights the important regions in an image that the AI model focuses on when making predictions.
        </AlertDescription>
      </Alert>

      {!data ? (
        <Card>
          <CardHeader>
            <CardTitle>No analysis available</CardTitle>
            <CardDescription>Run an analysis from the Detection page to view Grad-CAM visualizations.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
              <Brain className="h-12 w-12 mx-auto mb-2" />
              <div className="text-muted-foreground">Upload through Detection to see Grad-CAM</div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Original Image</CardTitle>
              <CardDescription>Uploaded leaf sample</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-square bg-muted rounded-lg relative overflow-hidden">
                {preview ? (
                  <img src={preview} alt="original" className="object-contain h-full w-full" />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                    <Brain className="h-12 w-12 mb-2" />
                    <div>No original preview available</div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Grad-CAM Overlay</CardTitle>
              <CardDescription>Model attention visualization</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="aspect-square relative rounded-lg overflow-hidden bg-muted">
                  {/* show overlay and heatmap */}
                  {data.gradcam_overlay_base64 ? (
                    <>
                      {/* If you have preview, overlay it on top with adjustable opacity */}
                      {preview ? (
                        <div className="h-full w-full relative">
                          <img src={preview} alt="orig" className="h-full w-full object-contain" />
                          <img
                            src={`data:image/png;base64,${data.gradcam_overlay_base64}`}
                            alt="overlay"
                            className="absolute left-0 top-0 h-full w-full object-contain"
                            style={{ opacity }}
                          />
                        </div>
                      ) : (
                        <img
                          src={`data:image/png;base64,${data.gradcam_overlay_base64}`}
                          alt="overlay"
                          className="h-full w-full object-contain"
                        />
                      )}
                    </>
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-muted-foreground">
                      No overlay image
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-4">
                  <label className="text-sm">Overlay opacity</label>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))}
                    className="w-full"
                  />
                </div>

                {data.description && (
                  <div>
                    <h4 className="font-semibold">Description</h4>
                    <p className="text-sm text-muted-foreground">{data.description}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
