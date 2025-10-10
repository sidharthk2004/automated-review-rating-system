import React from "react";
import type { NavigateFunction } from "react-router-dom";

import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Textarea,
  Label,
  Loading,
} from "../../../components";

import { Send } from "lucide-react";
import useReviewForm from "../hooks/useReviewForm";
import ShowRatingHidden from "./ShowRatingHidden";

const ReviewForm = ({ navigate }: { navigate: NavigateFunction }) => {
  const {
    body,
    onChangeBody,
    handleSubmit,
    isFill,
    isSubmitting,
    submitted,
    onChangeTitle,
    title,
    handleCancel,
    author,
    onChangeAuthor,
    isPending,
    setSubmitted,
    rating,
    errors,
  } = useReviewForm({ navigate });

  if (isPending) {
    return <Loading />;
  }

  return (
    <>
      <Card className=" border-0 shadow-none rounded-2xl overflow-hidden md:shadow-lg ">
        <CardHeader className="border-b border-slate-200">
          <CardTitle className="text-slate-800">Your Review</CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Author Name Input */}
            <div>
              <Label
                htmlFor="author"
                className="block text-sm font-medium text-slate-700 mb-2"
              >
                Author's Name
              </Label>
              <Input
                type="text"
                id="author"
                value={author}
                onChange={onChangeAuthor}
                placeholder="Enter your name..."
                className="w-full px-4 py-2 ring-0 rounded-lg border-blue-200 focus:ring-2 focus:ring-blue-300"
                required
              />
              {errors.author && (
                <span className="mt-1 block text-[12px] text-red-600">
                  {errors.author}
                </span>
              )}
            </div>
            {/* Title Input */}
            <div>
              <Label
                htmlFor="title"
                className="block text-sm font-medium text-slate-700 mb-2"
              >
                Review Title
              </Label>
              <Input
                type="text"
                id="title"
                value={title}
                onChange={onChangeTitle}
                placeholder="Summarize your opinion of this book"
                className="w-full px-4 py-2 rounded-lg ring-0 border-blue-200 focus:ring-2 focus:ring-blue-300"
                required
              />
               {errors.title && (
                <span className="mt-1 block text-[12px] text-red-600">
                  {errors.title}
                </span>
              )}
            </div>
           

            {/* Review Body */}
            <div>
              <Label
                htmlFor="body"
                className="block text-sm font-medium text-slate-700 mb-2"
              >
                Your Review
              </Label>
              <Textarea
                id="body"
                rows={2}
                maxLength={500}
                cols={2}
                value={body}
                onChange={onChangeBody}
                placeholder="Share your thoughts on this book..."
                className="w-full max-h-[150px] px-4 py-2  rounded-lg  resize-none border-blue-200 focus:ring-2 focus:ring-blue-300"
                required
              />
              {errors.body && (
                <span className="mt-1 block text-[12px] text-red-600">
                  {errors.body}
                </span>
              )}
            </div>

            {/* Submit Button */}
            <div className="pt-4 flex justify-end">
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
                className="mr-3"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={isSubmitting || !isFill}
                className="bg-gradient-to-r from-blue-600 to-cyan-500 text-white rounded-lg shadow-md hover:shadow-lg transition-all duration-300 min-w-[140px]"
              >
                {isSubmitting ? (
                  <div className="flex items-center justify-center">
                    <div className="w-4 h-4 border-t-2 border-r-2 border-white rounded-full animate-spin mr-2"></div>
                    Submitting...
                  </div>
                ) : (
                  <div className="flex items-center justify-center">
                    <Send size={16} className="mr-2" />
                    Submit
                  </div>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
      <ShowRatingHidden
        rating={rating}
        submitted={submitted}
        onClose={() => {
          setSubmitted(false);
        }}
      />
    </>
  );
};

export default React.memo(ReviewForm);
