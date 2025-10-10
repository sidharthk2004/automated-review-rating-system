import { type JSX } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
} from "../../../components";
import { ChevronRight } from "lucide-react";
import type { MockData } from "../../../lib";
import { Link } from "react-router-dom";
import AnimatedProgressBar from "./AnimatedProgressBar";

const TotalRatingCard = ({
  review,
  renderStars,
  totalCount,
}: {
  review: MockData;
  renderStars: (n: number) => JSX.Element;
  totalCount: number;
}) => {
  return (
    <div className="flex flex-col gap-4 max-w-82">
      <Card className="bg-gradient-to-br from-slate-50 to-slate-100 border-0 shadow-lg rounded-2xl transition-all duration-300 hover:shadow-xl">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-4">
            <div className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">
              {review.average}
            </div>
            <div>
              {renderStars(review.average)}
              <div className="text-sm text-slate-500 mt-1">
                {review.totalRatings} rating & reviews
              </div>
            </div>
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">
          {review.breakdown.map((b,key) => {
            const pct = Math.round((b.count / totalCount) * 100);
            return (
              <div key={key} className="flex items-center gap-3">
                <div className="w-10 text-sm font-medium">{b.stars}★</div>
                <div className="flex-1">
                  <AnimatedProgressBar value={pct} />
                </div>
                <div className="w-12 text-sm text-right font-medium">
                  {b.count}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* CTA button with animation */}
      <Link to={`/review-write`} className=" py-3 px-6 mx-auto">
        <Button className="mx-auto bg-gradient-to-r from-blue-600 to-cyan-500 border-0 text-white py-3 px-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
          Write a product review
          <ChevronRight size={16} className="ml-1" />
        </Button>
      </Link>
    </div>
  );
};

export default TotalRatingCard;
