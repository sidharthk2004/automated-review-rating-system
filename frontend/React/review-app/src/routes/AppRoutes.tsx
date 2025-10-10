import { BrowserRouter, Route, Routes } from "react-router-dom";
import MainLayout from "../layouts/MainLayout";
import { ErrorPage, ReviewPage, WriteReviewPage } from "../pages";



const AppRoutes = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<MainLayout />}>
          <Route path="/" element={<ReviewPage />} />
          <Route path="/review-write" element={<WriteReviewPage />} />
          <Route path="*" element={<ErrorPage />} />

        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default AppRoutes;
