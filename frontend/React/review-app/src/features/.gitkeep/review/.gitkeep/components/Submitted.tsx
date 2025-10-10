import { Button } from "../../../components";
import type { NavigateFunction } from "react-router-dom";

export default function Submitted({
  navigate,
}: {
  navigate: NavigateFunction;
}) {
  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="text-center py-12">
        <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
          <svg
            className="w-10 h-10 text-green-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 13l4 4L19 7"
            />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-slate-800 mb-2">
          Review Submitted!
        </h2>
        <p className="text-slate-600 mb-6">
          Thank you for sharing your thoughts.
        </p>
        <Button
          onClick={() => navigate(-1)}
          className="bg-gradient-to-r from-blue-600 to-cyan-500"
        >
          Return to Book
        </Button>
      </div>
    </div>
  );
}
