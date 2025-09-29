// Loader.jsx remains the same as the previous, modern version.
import React from "react";

function Loader() {
  return (
    <div className="mt-12 flex flex-col items-center">
      <div className="
        animate-spin 
        rounded-full 
        border-4 
        border-[color:var(--color-surface)] 
        border-t-[color:var(--color-primary)] 
        h-16 w-16 mb-4
      "></div>
      <p className="text-[color:var(--color-primary)] font-semibold text-lg animate-pulse">
        Analyzing Leaf Structure...
      </p>
    </div>
  );
}

export default Loader;