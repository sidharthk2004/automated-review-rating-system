import { useReviews } from "../hooks/useShowReview";
import { ChevronRight, MessageSquareOff } from "lucide-react";
import {
  Badge,
  Button,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../components";
import TotalRatingCard from "../components/TotalRatingCard";
import ReviewCard from "../components/ReviewCard";
import renderStars from "../components/RenderStars";
import { Loading } from "../../../components";
import { Link } from "react-router-dom";
export default function ShowReviewPage() {
  const {
    resData,
    reviews,
    loading,
    error,
    sortOption,
    setSortOption,
    displayedReviews,
    hasMoreReviews,
    toggleShowAll,
  } = useReviews({ initialVisible: 3, initialSort: "newest" });



  if (loading) {
    return <Loading />;
  }
  if (error) {
    return <div className="text-center p-10 text-red-500">{error}</div>;
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="sm:text-[25px] font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
          Ratings & Reviews
        </h1>
        {resData?.totalRatings && (
          <Badge
            variant="secondary"
            className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full"
          >
            {resData?.totalRatings} ratings
          </Badge>
        )}
      </div>

      {resData?.breakdown?.length ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <TotalRatingCard
            review={resData!}
            totalCount={resData.totalRatings}
            renderStars={renderStars}
          />

          <div className="md:col-span-2 space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-lg font-semibold">Customer Reviews</h2>
              <div className="relative max-w-[120px] hover:ring-0 focus:ring-0 focus:ring-offset-0">
                <Select value={sortOption} onValueChange={setSortOption}>
                  <SelectTrigger
                    size="sm"
                    className="h-6 px-2 bg-blue-400 text-white text-[12px] rounded-lg focus:ring-0 focus:ring-offset-0 [&>svg]:stroke-white [&>svg]:text-white"
                  >
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent className="bg-blue-400 text-white border-none">
                    <SelectItem value="newest">Newest first</SelectItem>
                    <SelectItem value="oldest">Oldest first</SelectItem>
                    <SelectItem value="highest">Highest rated</SelectItem>
                    <SelectItem value="lowest">Lowest rated</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {displayedReviews ? (
              displayedReviews.map((rev, index) => (
                <ReviewCard
                  key={index ?? `${rev.createdAt ?? "no-date"}-${index}`}
                  renderStars={renderStars}
                  rev={rev}
                />
              ))
            ) : (
              <div className="flex flex-col items-center justify-center py-10 text-center text-gray-500">
                <MessageSquareOff className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">No reviews yet</p>
                <p className="text-sm text-gray-400">
                  Be the first to share your thoughts!
                </p>
              </div>
            )}

            {(reviews?.length ?? 0) > 3 && (
              <div className="flex justify-center mt-6">
                <Button
                  onClick={toggleShowAll}
                  className="py-0 my-0 bg-blue-400 text-white text-[12px] rounded-[16px] shadow-gray-300 hover:bg-blue-500 transition-colors"
                >
                  {hasMoreReviews ? "View All Reviews" : "Show Less"}
                </Button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-10 text-center text-gray-500 backdrop-blur-3xl">
          <MessageSquareOff className="w-12 h-12 mb-3 text-gray-400" />
          <p className="text-lg font-medium">No reviews yet...</p>
          <p className="text-sm text-gray-400">
            Be the first to share your thoughts!
          </p>
          <Link to={`/review-write`} className=" py-3 px-3 ">
            <Button className="text-[12px] bg-gradient-to-r from-gray-400 to-slate-500 border-0 text-white rounded-xl shadow-md hover:shadow-lg transition-all duration-100 transform hover:-translate-y-1">
              Write a product review
              <ChevronRight size={16} className="ml-1" />
            </Button>
          </Link>
        </div>
      )}
    </div>
  );
}
