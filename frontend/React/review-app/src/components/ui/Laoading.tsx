import React from "react";

type LoadingProps = {
  message?: string;
  size?: number; // px
  fullScreen?: boolean;
  className?: string;
};

export default function Loading({
  message = "Loading…",
  size = 48,
  fullScreen = true,
  className = "",
}: LoadingProps) {
  const containerClass = fullScreen
    ? "fixed inset-0 z-50 flex items-center justify-center bg-white/60 backdrop-blur-sm"
    : "flex items-center justify-center";

  // inline style for svg size so it scales precisely
  const svgStyle: React.CSSProperties = { width: size, height: size };

  return (
    <div className={`${containerClass} ${className}`} aria-busy="true" aria-live="polite">
      <div className="flex flex-col items-center gap-3 px-4">
        {/* Role=status required for assistive tech; motion-safe ensures reduced-motion users aren't spun */}
        <div role="status" className="flex items-center gap-3">
          <svg
            style={svgStyle}
            className="text-blue-600 motion-safe:animate-spin motion-reduce:animate-none"
            viewBox="0 0 50 50"
            fill="none"
            aria-hidden="true"
          >
            <circle
              cx="25"
              cy="25"
              r="20"
              stroke="currentColor"
              strokeOpacity="0.12"
              strokeWidth="6"
            />
            <path
              d="M45 25a20 20 0 00-7.7-15.4"
              stroke="currentColor"
              strokeWidth="6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>

          {/* Visible message for sighted users */}
          <span className="text-sm font-medium text-slate-700">{message}</span>

          {/* Hidden text for screen readers (keeps announcements concise) */}
          <span className="sr-only">{message}</span>
        </div>

        {/* Optional tiny helper/subtext */}
        <p className="text-xs text-slate-500 max-w-xs text-center leading-snug">
          Please wait a moment.
        </p>
      </div>
    </div>
  );
}
