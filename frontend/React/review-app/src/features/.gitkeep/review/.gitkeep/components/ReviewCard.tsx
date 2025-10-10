import { useState, type JSX } from "react";
import {
  Card,
  Avatar,
  AvatarFallback,
  CardHeader,
} from "../../../components";
import type { Review } from "../../../lib";
import { formatTimeAgo } from "../../../utils";
import useInitials from "../../../hooks/useInitials";

const ReviewCard = ({
  rev,
  renderStars,
}: {
  rev: Review;
  renderStars: (n: number) => JSX.Element;
}) => {
  const { initials } = useInitials({ author: rev.author });
  const [expanded, setExpanded] = useState(false);

  if (!rev) return null;

  return (
    <Card className="border-0 shadow-sm rounded-md transition-all hover:shadow-md focus-within:shadow-md">
      <CardHeader className=" py-0 px-3 pt-0">
        <div className="flex items-start gap-2">
          <Avatar className="w-6 h-6 border border-gray-200">
            <AvatarFallback className="bg-gray-100 text-xs text-gray-600">
              {initials}
            </AvatarFallback>
          </Avatar>

          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-1 justify-between">
              <div className="font-medium text-xs text-slate-800">
                {rev.title}
              </div>
              <div className="text-xs text-slate-400 pr-1">
                {formatTimeAgo(rev.createdAt)}
              </div>
            </div>

            <div className="flex items-center gap-1 mt-0.5">
              {renderStars(rev.rating)}
              <span className="text-xs text-slate-500 font-medium">
                {rev.rating}/5
              </span>
            </div>

            <p
              className={`text-xs text-slate-600 leading-relaxed pt-2 ${
                expanded ? "" : "line-clamp-3 "
              }`}
            >
              {rev.body}
            </p>
            {rev.body && rev.body.length > 100 && (
              <button
                onClick={() => setExpanded((prev) => !prev)}
                className="mt-1 text-gray-600 text-xs font-medium hover:underline"
              >
                {expanded ? "Show less" : "Read more"}
              </button>
            )}
          </div>
        </div>
      </CardHeader>
    </Card>
  );
};

export default ReviewCard;
