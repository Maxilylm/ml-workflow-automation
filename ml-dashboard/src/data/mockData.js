export const mockMetrics = {
  accuracy: 0.94,
  precision: 0.92,
  recall: 0.89,
  f1: 0.90,
  createdAt: '2026-02-17T10:00:00Z'
};

export const feature3Options = [1, 2, 3];

export const simulatePrediction = (f1, f2, f3) => {
  // Simple formula for demo purposes
  const result = f1 * 0.3 + f2 * 0.5 + f3 * 10;
  return result.toFixed(2);
};
