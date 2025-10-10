import { useEffect, useState } from "react";

const AnimatedProgressBar = ({ value }: { value: number }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setProgress(value);
    }, 300);
    return () => clearTimeout(timer);
  }, [value]);

  return (
    <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
      <div  
        className={`h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-1000 ease-out`}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
};
export default AnimatedProgressBar;
