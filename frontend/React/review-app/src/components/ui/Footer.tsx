import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer
      className="
        bg-black w-full z-50 text-slate-200
        pt-4 pb-[calc(env(safe-area-inset-bottom,0px)+24px)]
        md:pt-6 md:pb-[calc(env(safe-area-inset-bottom,0px)+32px)]
      "
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* top area: logo + links */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-start sm:items-center gap-3">
            <Link to="/" className="inline-block">
              <span className="text-white text-lg font-semibold">LOGO</span>
            </Link>
            <p className="text-sm text-slate-400 hidden sm:block">
              Review your favorite books
            </p>
          </div>

         
        </div>

        {/* bottom area: copyright */}
        <div className="mt-4 border-t border-slate-800 pt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-sm text-slate-500">
          <p>© {new Date().getFullYear()} Your Company. All rights reserved.</p>
          <div className="text-xs text-slate-500">Made with care.</div>
        </div>
      </div>
    </footer>
  );
}