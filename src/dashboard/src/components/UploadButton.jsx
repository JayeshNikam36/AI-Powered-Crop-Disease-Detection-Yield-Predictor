import React from "react";
import { Leaf } from "lucide-react"; 

function UploadButton({ onChange }) {
  return (
    <label 
        className="
            flex flex-col items-center justify-center 
            w-full max-w-2xl h-64 
            border-2 border-dashed border-slate-600 
            rounded-3xl cursor-pointer 
            bg-[color:var(--color-surface)] 
            shadow-xl shadow-[rgba(0,0,0,0.4)]
            hover:bg-slate-700/50 
            hover:border-[color:var(--color-primary)]
            transition duration-300 ease-in-out
        "
    >
      <Leaf className="w-16 h-16 text-[color:var(--color-primary)] mb-4" /> 
      <span className="text-xl text-[color:var(--color-text-light)] font-bold mb-1">
        Click to Upload
      </span>
      <span className="text-sm text-[color:var(--color-text-muted)]">
        or drag & drop your image file (PNG, JPG)
      </span>
      <input
        type="file"
        accept="image/*"
        className="hidden"
        // Changed from handleUpload to onChange which now only calls handleFileChange
        onChange={onChange} 
      />
    </label>
  );
}

export default UploadButton;