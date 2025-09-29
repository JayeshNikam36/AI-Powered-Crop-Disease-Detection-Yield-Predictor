import React from "react";

export function Button({ children, onClick, disabled, variant = "primary" }) {
    let baseClasses = "px-6 py-3 rounded-xl font-bold transition-all duration-200 flex items-center justify-center";
    
    // Define styles based on the 'variant' prop
    if (variant === "primary") {
        baseClasses += " bg-[color:var(--color-primary)] text-slate-900 shadow-lg hover:bg-emerald-400 hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed";
    } else if (variant === "secondary") {
        baseClasses += " bg-slate-700 text-[color:var(--color-text-light)] border border-slate-600 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed";
    } else if (variant === "danger") {
        baseClasses += " bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed";
    }

    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={baseClasses}
        >
            {children}
        </button>
    );
}