import React from "react";

// Use a border for a cleaner, modern edge instead of a heavy shadow
// Apply background and border colors from the new dark theme
export function Card({ children, className }) {
  return (
    <div
      className={`
        rounded-xl 
        bg-[color:var(--color-surface)] 
        border border-slate-700 
        shadow-2xl shadow-[rgba(0,0,0,0.3)] 
        transition-all duration-300
        ${className || ""}
      `}
    >
      {children}
    </div>
  );
}

// Keep CardContent clean but adjust padding for the modern look
export function CardContent({ children, className }) {
  return (
    <div className={`p-6 ${className || ""}`}>
      {children}
    </div>
  );
}