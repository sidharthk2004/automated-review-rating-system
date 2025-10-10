import { Outlet } from "react-router-dom";



const MainLayout = () => {
  return (
    // full viewport, column layout
    <div className="min-h-screen flex flex-col w-full">


      {/* main content grows to take remaining space */}
      <main className="flex-1 w-full p-4">
        <Outlet />
      </main>

    </div>
  );
};

export default MainLayout;