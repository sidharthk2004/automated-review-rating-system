export type RatingBreakdown = {
  stars: number;
  count: number;
};

export type Review = {
  id: number;
  author: string;
  title: string;
  body: string;
  rating: number;
  createdAt: Date;
  modifiedAt?: Date;
};

export type MockData = {
  average: number;
  totalRatings: number;
  breakdown: RatingBreakdown[];
  reviews: Review[];
};

export type CreateReviewType = {
  author: string;
  title: string;
  body: string;
};

export type ValidationResult = {
  valid: boolean;
  reason?: string;
};