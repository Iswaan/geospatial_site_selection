/**
 * Formats a number as Indian Rupees using the Indian numbering system.
 * Uses Intl.NumberFormat which is safe to call client-side only.
 * e.g. 4089123 → "₹40,89,123"
 *      -3867   → "-₹3,867"
 */
export function formatINR(value: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(value);
}
