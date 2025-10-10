import { Star } from "lucide-react";
import type { JSX } from "react";

export default function renderStars(n: number): JSX.Element {
    const filled = Math.round(n);
    return (
      <div className="flex items-center gap-1">
        {Array.from({ length: 5 }).map((_, i) => (
          <Star
            key={i}
            size={16}
            fill={i < filled ? "currentColor" : "none"}
            className={i < filled ? "text-yellow-400" : "text-slate-300"}
          />
        ))}
      </div>
    );
  }