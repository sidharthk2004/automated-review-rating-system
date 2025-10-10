// ShowRatingHidden.tsx
import React from "react";
import renderStars from "./RenderStars";
import { Link } from "react-router-dom";
import { Button } from "../../../components";
import { ChevronRight } from "lucide-react";

type Props = {
  submitted: boolean;
  rating: number;
  onClose?: () => void;
};

const ShowRatingHidden: React.FC<Props> = ({ submitted, rating, onClose }) => {
  if (!submitted) return null;

  const handleClose = () => {
    if (onClose) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="rating-success-heading"
    >
      {/* Backdrop with click handler */}
      <div 
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={handleClose}
        aria-hidden="true"
      />
      
      {/* Card */}
      <div
        className="relative z-10 w-full max-w-sm rounded-2xl bg-white/95 dark:bg-slate-900/95
                   shadow-2xl border border-white/30 dark:border-slate-800/60
                   p-6 text-center transform transition-all duration-200 ease-out
                   animate-in fade-in-90 zoom-in-90"
      >
        <h3 
          id="rating-success-heading"
          className="text-lg font-semibold mb-1 text-slate-800 dark:text-slate-100"
        >
          Thanks — review submitted!
        </h3>

        <p className="text-sm text-slate-500 dark:text-slate-300 mb-4">
          We received your review and rating.
        </p>

        <div className="flex items-center justify-center gap-4 mb-3">
          <div className="text-3xl font-extrabold text-sky-600 dark:text-sky-400">
            {rating.toFixed(0)}
          </div>

          <div
            className="text-xl tracking-wider text-yellow-500 dark:text-yellow-400"
            aria-label={`Rating: ${rating} out of 5 stars`}
          >
            {renderStars(rating)}
          </div>
        </div>

        <div className="mt-4">
          <Link to="/" onClick={handleClose}>
            <Button 
              className="text-sm bg-gradient-to-bl from-blue-400 to-cyan-500 border-0 
                         text-white rounded-xl shadow-md hover:shadow-lg transition-all 
                         duration-100 transform hover:-translate-y-0.5"
            >
              Go back
              <ChevronRight size={16} className="ml-1" />
            </Button>
          </Link> 
        </div>
        
        {/* Close button for better accessibility */}
        <button
          onClick={handleClose}
          className="absolute top-4 right-4 p-1 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800"
          aria-label="Close dialog"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ShowRatingHidden;