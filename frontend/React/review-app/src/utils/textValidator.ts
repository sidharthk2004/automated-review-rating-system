import type { ValidationResult } from "../lib/types";


/**
 * Validate author name
 */
export function isValidAuthor(value: string): ValidationResult {
  const v = value.trim();

  if (v.length === 0) return { valid: false, reason: "Author is required." };
  if (v.length < 2) return { valid: false, reason: "Name must be at least 2 characters." };
  if (v.length > 50) return { valid: false, reason: "Name must be 50 characters or less." };

  // Only letters + spaces
  const lettersOnly = /^[A-Za-z\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\s]+$/u;
  if (!lettersOnly.test(v)) {
    return { valid: false, reason: "Name can only contain letters and spaces." };
  }

  //  Must have at least one letter
  const containsLetter = /[A-Za-z\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF]/u;
  if (!containsLetter.test(v)) {
    return { valid: false, reason: "Name must contain at least one letter." };
  }

  //  Must have at least one vowel (English vowels)
  const containsVowel = /[aeiouAEIOU]/;
  if (!containsVowel.test(v)) {
    return { valid: false, reason: "Name look unusual" };
  }

  // Reject repeated single character like "aaaaa"
  if (/^(.)\1+$/.test(v)) {
    return { valid: false, reason: "Name looks invalid (repeated characters)." };
  }

  return { valid: true };
}


/**
 * Validate title
 */
export function isValidTitle(value: string): ValidationResult {
  const v = value.trim();

  if (v.length === 0) return { valid: false, reason: "Title is required." };
  if (v.length < 5) return { valid: false, reason: "Title must be at least 5 characters." };
  if (v.length > 100) return { valid: false, reason: "Title must be 100 characters or less." };

  const lettersOnly = /^[A-Za-z\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\s]+$/u;
  if (!lettersOnly.test(v)) {
    return { valid: false, reason: "Title can only contain letters and spaces." };
  }

  const containsLetter = /[A-Za-z\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF]/u;
  if (!containsLetter.test(v)) {
    return { valid: false, reason: "Title must contain at least one letter." };
  }
    // Must contain at least one vowel (or meaningful character for CJK)
  const hasVowelOrCJK = /[aeiouAEIOU\u3040-\u30ff\u4e00-\u9fff]/u;
  if (!hasVowelOrCJK.test(v)) {
    return { valid: false, reason: "Review looks unusual — include some words." };
  }

  // Reject trivial titles
  const trivialTitles = /^(ab+c|abcde|abcd|xyz|test|title|lorem)$/i;
  if (trivialTitles.test(v)) {
    return { valid: false, reason: "Please provide a more descriptive title." };
  }

  // Reject repeated single character
  if (/^(.)\1+$/.test(v)) {
    return { valid: false, reason: "Title looks invalid (repeated characters)." };
  }

  return { valid: true };
}

/**
 * Validate review body
 */
export function isMeaningfulBody(value: string): ValidationResult {
  const body = value.trim();

  if (body.length < 10) {
    return { valid: false, reason: "Review must be at least 10 characters." };
  }

  // Must contain at least one vowel (or meaningful character for CJK)
  const hasVowelOrCJK = /[aeiouAEIOU\u3040-\u30ff\u4e00-\u9fff]/u;
  if (!hasVowelOrCJK.test(body)) {
    return { valid: false, reason: "Review looks unusual — include some words." };
  }

  // Allow letters, digits, spaces, and period
  const invalidChars = /[^0-9A-Za-z\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\s.\u3002]/u;
  if (invalidChars.test(body)) {
    return { valid: false, reason: "Review contains invalid characters. Only letters, digits, spaces and '.' are allowed." };
  }

  if (/^(.)\1+$/.test(body)) {
    return { valid: false, reason: "Review looks like repeated characters." };
  }

  if (/(abcde|abcd|abcdef|12345|qwerty|asdf)/i.test(body)) {
    return { valid: false, reason: "Review looks like a placeholder or sequential text." };
  }

  // Encourage proper sentence structure for short reviews
  const hasWhitespace = /\s/.test(body);
  const hasPeriod = /[.\u3002]/.test(body);
  if ((!hasWhitespace || !hasPeriod) && body.length < 40) {
    return { valid: false, reason: "Try writing a little more — include sentences and use periods." };
  }

  // Check word repetition ratio
  const normalized = body.toLowerCase().replace(/[.\u3002]+/g, "");
  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length > 3) {
    const wordCounts = words.reduce<Record<string, number>>((acc, w) => {
      acc[w] = (acc[w] || 0) + 1;
      return acc;
    }, {});
    const [[topWord, topCount]] = Object.entries(wordCounts).sort((a, b) => b[1] - a[1]);
    if (topCount / words.length > 0.6) {
      return { valid: false, reason: "Try to add more variety to your review." };
    }
  }

  return { valid: true };
}

/**
 * Validate entire review object
 */
export function validateReview(input: { author: string; title: string; body: string }) {
  const author = isValidAuthor(input.author);
  const title = isValidTitle(input.title);
  const body = isMeaningfulBody(input.body);

  return {
    author,
    title,
    body,
    valid: author.valid && title.valid && body.valid,
  };
}
