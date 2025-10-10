import { useEffect, useMemo, useState } from "react";
import type { MockData, Review } from "../../../lib";
import { getReviews } from "../../../lib/axios/reviewInstance";

/**
 * useReviews - custom hook to encapsulate fetching, sorting and pagination
 * of reviews so the ShowReviewPage component is much slimmer.
 */
export function useReviews({ initialVisible = 3, initialSort = "newest" } = {}) {
  const [resData, setResData] = useState<MockData | undefined>();
  const [reviews, setReviews] = useState<Review[]>([]);
  const [visibleReviews, setVisibleReviews] = useState<number>(initialVisible);
  const [sortOption, setSortOption] = useState<string>(initialSort);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await getReviews();
        console.log(res.data)
        if (!res?.data) throw new Error("Invalid response");
        if (!mounted) return;
        setResData(res.data as MockData);
        setReviews(Array.isArray(res.data.reviews) ? res.data.reviews : []);
      } catch (err) {
        console.error(err);
        if (mounted) setError("Failed to load reviews");
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);
  const totalCount = resData?.breakdown?.reduce((s, r) => s + r.count, 0) || 1;

  // memoize sorted list so we don't sort on every render
  const sortedReviews = useMemo(() => {
    const arr = [...reviews];
    switch (sortOption) {
      case "newest":
        return arr.sort(
          (a, b) => new Date(b.createdAt || "").getTime() - new Date(a.createdAt || "").getTime()
        );
      case "oldest":
        return arr.sort(
          (a, b) => new Date(a.createdAt || "").getTime() - new Date(b.createdAt || "").getTime()
        );
      case "highest":
        return arr.sort((a, b) => b.rating - a.rating);
      case "lowest":
        return arr.sort((a, b) => a.rating - b.rating);
      default:
        return arr;
    }
  }, [reviews, sortOption]);

  const displayedReviews = useMemo(
    () => sortedReviews.slice(0, visibleReviews),
    [sortedReviews, visibleReviews]
  );

  const hasMoreReviews = visibleReviews < (reviews?.length ?? 0);

  // convenience helpers
  function showAll() {
    setVisibleReviews(reviews.length);
  }
  function showLess() {
    setVisibleReviews(initialVisible);
  }
  function toggleShowAll() {
    hasMoreReviews ? showAll() : showLess();
  }

  return {
    resData,
    reviews,
    loading,
    error,
    visibleReviews,
    setVisibleReviews,
    sortOption,
    setSortOption,
    sortedReviews,
    displayedReviews,
    hasMoreReviews,
    showAll,
    showLess,
    toggleShowAll,
    totalCount,

  } as const;
}
