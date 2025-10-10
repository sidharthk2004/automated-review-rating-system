import { useMemo } from "react";

export default function useInitials({ author }: { author: string }) {
  const initials = useMemo(() => {
    return (author ?? "")
      .split(" ")
      .map((n) => (n ? n[0] : ""))
      .slice(0, 2)
      .join("")
      .toUpperCase();
  }, [author]);

  return { initials };
}
