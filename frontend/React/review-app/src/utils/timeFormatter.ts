// utils/timeFormatter.ts
type InputDate =  | Date  | string  | number  | null  | undefined  | { toDate?: () => Date; seconds?: number };

const toDate = (input?: InputDate): Date | null => {
  if (input == null) return null;
  if (input instanceof Date) return input;
  const d = new Date(input as string | number);
  return isNaN(d.getTime()) ? null : d;
};

export const formatTimeAgo = (input?: InputDate): string => {
  const date = toDate(input);
  if (!date) return "";

  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 5) return "just now"; // handle very recent times

  const units: [number, string][] = [
    [31536000, "year"],
    [2592000, "month"],
    [604800, "week"],
    [86400, "day"],
    [3600, "hour"],
    [60, "minute"],
    [1, "second"],
  ];
  for (const [secsPerUnit, name] of units) {
    const count = Math.floor(seconds / secsPerUnit);
    if (count > 0) return `${count} ${name}${count > 1 ? "s" : ""} ago`;
  }
  return "just now";
};
