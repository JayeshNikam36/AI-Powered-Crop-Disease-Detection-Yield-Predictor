// src/dashboard/src/App.jsx

import React, { useState } from "react";
import PredictionResult from "./components/PredictionResult";
import Loader from "./components/Loader";
import UploadButton from "./components/UploadButton";
import { Button } from "./ui/Button";
// --- THE CRITICAL FIX IS HERE ---
import { Zap } from "lucide-react"; 
// --------------------------------

function App() {
  const [file, setFile] = useState(null); 
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile); 
    setImage(URL.createObjectURL(selectedFile)); 
    setPrediction(null);
  };

  const handlePredict = async () => {
    if (!file) {
      alert("Please upload an image first.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Prediction failed");
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };
  
  const resetApp = () => {
      setFile(null);
      setImage(null);
      setPrediction(null);
      setLoading(false);
  }

  return (
    <div className="min-h-screen bg-[color:var(--color-background)] flex flex-col items-center">
      <div className="modern-container">
        <header className="flex justify-between items-center py-6 border-b border-slate-700 mb-8 w-full max-w-7xl">
            <h1 className="text-2xl font-extrabold tracking-tight text-[color:var(--color-primary)]">
                AgriScan
            </h1>
            {prediction && (
                <Button onClick={resetApp} variant="secondary">
                    New Analysis
                </Button>
            )}
        </header>

        {!prediction ? (
          <div className="flex flex-col items-center justify-center text-center px-6 py-10 max-w-3xl mx-auto">
            <h1 className="text-5xl font-extrabold text-[color:var(--color-text-light)] mb-4 tracking-tight">
              AI-Powered <span className="text-[color:var(--color-primary)]">Crop Health</span> Analyzer
            </h1>
            <p className="text-xl text-[color:var(--color-text-muted)] mb-4">
              Upload a photo of your crop leaf to instantly diagnose diseases and visualize model focus.
            </p>
            
            <div className="text-left bg-[color:var(--color-surface)] border border-slate-700 p-6 rounded-xl shadow-inner my-8 w-full">
                <h2 className="text-2xl font-semibold text-[color:var(--color-primary)] mb-3">Project Overview</h2>
                <p className="text-base text-[color:var(--color-text-light)] leading-relaxed">
                    This tool utilizes a cutting-edge **Convolutional Neural Network (CNN)** to perform rapid, high-accuracy classification of common crop leaf diseases. Beyond just a prediction, we employ **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate visual heatmaps. These heatmaps pinpoint the exact regions of the leaf the model focused on to make its decision, providing crucial **model interpretability** and verifying the diagnosis.
                </p>
                <ul className="list-disc list-inside text-sm text-[color:var(--color-text-muted)] mt-4 space-y-1">
                    <li>**Input:** Crop leaf image (e.g., tomato, potato, corn).</li>
                    <li>**Output:** Predicted disease class and confidence score.</li>
                    <li>**Key Feature:** Visual evidence (Grad-CAM) for diagnostic verification.</li>
                </ul>
            </div>
            
            <UploadButton onChange={handleFileChange} />

            {image && !loading && (
              <div className="mt-8 flex flex-col items-center w-full">
                <p className="text-sm font-medium text-[color:var(--color-text-muted)] mb-3">
                    Selected Image Preview:
                </p>
                <img
                  src={image}
                  alt="Preview"
                  className="w-64 h-64 object-cover rounded-xl shadow-2xl border border-slate-700 mb-6"
                />
                
                <Button onClick={handlePredict} disabled={!file} variant="primary">
                    <Zap className="w-5 h-5 mr-2" />
                    Run AI Prediction
                </Button>
              </div>
            )}

            {loading && <Loader />}
          </div>
        ) : (
          <PredictionResult prediction={prediction} uploadedImage={image} />
        )}
      </div>
    </div>
  );
}

export default App;