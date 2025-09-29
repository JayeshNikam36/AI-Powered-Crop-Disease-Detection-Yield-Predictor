import React from "react";
import { Card, CardContent } from "../ui/Card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Zap, Activity, Repeat2 } from "lucide-react";
import { Button } from "../ui/Button";

// Custom Tooltip for the chart (from previous step - no major changes needed)
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="p-3 bg-slate-700/95 border border-slate-600 rounded-lg shadow-lg">
        <p className="text-sm font-semibold text-[color:var(--color-primary)]">{label}</p>
        <p className="text-lg text-[color:var(--color-text-light)]">{`${payload[0].value}%`}</p>
      </div>
    );
  }
  return null;
};

function PredictionResult({ prediction, uploadedImage }) {
  const probs = prediction.probabilities.map((p, i) => ({
    name: `Class ${i}`,
    value: parseFloat((p * 100).toFixed(2)),
  }));

  return (
    <div className="modern-container">
      <h2 className="text-4xl font-extrabold text-[color:var(--color-text-light)] mb-12 text-center tracking-tight">
        Analysis Complete <span className="text-[color:var(--color-primary)]">🔬</span>
      </h2>

      {/* Main Grid for Alignment: 2-column layout for visual content, 1 column for data/chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Col: Prediction Summary & Chart (1 column wide) */}
        <div className="lg:col-span-1 space-y-8 flex flex-col">
          <Card className="shadow-2xl shadow-[rgba(16,185,129,0.15)] border-l-4 border-[color:var(--color-primary)] flex-shrink-0">
            <CardContent className="p-6">
              <div className="flex items-center space-x-3 mb-4 border-b border-slate-700 pb-2">
                <Zap className="w-6 h-6 text-[color:var(--color-primary)]" />
                <h3 className="text-2xl font-bold text-[color:var(--color-text-light)]">
                  Prediction Summary
                </h3>
              </div>
              <p className="text-lg text-[color:var(--color-text-muted)] mt-4">
                <span className="font-semibold text-[color:var(--color-text-light)]">Detected Class:</span>{" "}
                <span className="text-xl font-extrabold text-[color:var(--color-primary)] block mt-1">
                    {prediction.predicted_class}
                </span>
              </p>
              <p className="text-lg text-[color:var(--color-text-muted)] mt-4">
                <span className="font-semibold text-[color:var(--color-text-light)]">Confidence Score:</span>{" "}
                <span className="text-xl font-extrabold text-[color:var(--color-accent)] block mt-1">
                  {(prediction.score * 100).toFixed(2)}%
                </span>
              </p>
            </CardContent>
          </Card>
          
          <Card className="flex-grow"> {/* Allows the chart card to fill remaining vertical space */}
            <CardContent className="p-6 h-full flex flex-col">
                <div className="flex items-center space-x-3 mb-4 border-b border-slate-700 pb-2">
                    <Activity className="w-6 h-6 text-[color:var(--color-accent)]" />
                    <h3 className="text-2xl font-bold text-[color:var(--color-text-light)]">
                        Class Probabilities
                    </h3>
                </div>
                {/* Chart with refined colors and typography */}
                <div className="w-full flex-grow pt-4 h-full min-h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={probs} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                            <XAxis 
                                type="number" 
                                stroke="var(--color-text-muted)" 
                                className="text-sm"
                                tickFormatter={(tick) => `${tick}%`}
                            />
                            <YAxis 
                                dataKey="name" 
                                type="category" 
                                stroke="var(--color-text-muted)" 
                                className="text-sm"
                            />
                            <CustomTooltip />
                            <Bar 
                                dataKey="value" 
                                fill="var(--color-accent)" 
                                radius={[4, 4, 0, 0]}                             
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
          </Card>
        </div>


        {/* Right Col: Image Visualizations (2 columns wide, split into a 1x1 image and 2 smaller images) */}
        <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Full width card for the main image */}
            <Card className="md:col-span-2 border-t-4 border-slate-700">
                <CardContent className="p-6">
                    <h3 className="text-xl font-bold text-[color:var(--color-text-light)] mb-4 border-b border-slate-700 pb-2">
                        Uploaded Image
                    </h3>
                    <img
                        src={uploadedImage}
                        alt="Uploaded"
                        // Added max-h for better visual balance with other images
                        className="rounded-lg w-full object-cover shadow-xl border border-slate-700 max-h-[350px] mx-auto"
                    />
                </CardContent>
            </Card>

            {/* Side-by-side cards for Grad-CAM */}
            <Card className="border-t-4 border-slate-700">
                <CardContent className="p-6">
                    <h3 className="text-xl font-bold text-[color:var(--color-text-light)] mb-4 border-b border-slate-700 pb-2">
                        Grad-CAM Overlay
                    </h3>
                    <img
                        src={`data:image/png;base64,${prediction.gradcam_overlay_base64}`}
                        alt="Grad-CAM Overlay"
                        className="rounded-lg w-full object-cover shadow-xl border border-slate-700"
                    />
                </CardContent>
            </Card>

            <Card className="border-t-4 border-slate-700">
                <CardContent className="p-6">
                    <h3 className="text-xl font-bold text-[color:var(--color-text-light)] mb-4 border-b border-slate-700 pb-2">
                        Grad-CAM Heatmap
                    </h3>
                    <img
                        src={`data:image/png;base64,${prediction.gradcam_heatmap_base64}`}
                        alt="Grad-CAM Heatmap"
                        className="rounded-lg w-full object-cover shadow-xl border border-slate-700"
                    />
                </CardContent>
            </Card>
        </div>
      </div>
    </div>
  );
}

export default PredictionResult;