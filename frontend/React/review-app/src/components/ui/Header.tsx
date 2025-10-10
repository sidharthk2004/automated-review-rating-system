import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "./button";

export default function Header() {
  const [open, setOpen] = useState(false);

  return (
    <header className={"w-full border-b bg-white relative mx-auto"}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center text-slate-900">
              <span className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">
                LOGO
              </span>
            </Link>
          </div>

          <nav className="hidden md:flex items-center gap-6">
            <Link to="/review" className="text-sm font-medium text-slate-700 hover:text-slate-900 transition-colors">
              Review
            </Link>

            <Link to="/login" className="inline-flex">
              <Button className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white border-0 shadow-md hover:shadow-lg transition-all">
                Login
              </Button>
            </Link>

            <Link to="/sign-up" className="inline-flex">
              <Button className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-600 text-white border-0 shadow-md hover:shadow-lg transition-all">
                Sign Up
              </Button>
            </Link>
          </nav>

          <div className="md:hidden flex items-center">
            <button
              onClick={() => setOpen((s) => !s)}
              aria-expanded={open}
              aria-label={open ? "Close menu" : "Open menu"}
              className="p-2 rounded-md inline-flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all"
            >
              {open ? (
                // X icon
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                // Hamburger icon
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {open && (
        <div className="md:hidden bg-white border-t animate-fadeIn">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 space-y-2">
            <Link to="/review" className="block text-base font-medium text-slate-700 hover:text-slate-900 transition-colors">
              Review
            </Link>

            <div className="grid grid-cols-2 gap-2">
              <Link to="/login" className="block w-full">
                <Button className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white border-0">
                  Login
                </Button>
              </Link>

              <Link to="/sign-up" className="block w-full">
                <Button className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white border-0">
                  Sign Up
                </Button>
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}