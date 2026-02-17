function PredictionResult({ result, error }) {
  if (error) {
    return (
      <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded">
        <p className="text-red-400">{error}</p>
      </div>
    );
  }

  if (result === null) {
    return null;
  }

  return (
    <div className="mt-4 p-3 bg-green-900/50 border border-green-700 rounded">
      <p className="text-gray-300 text-sm">Prediction Result</p>
      <p className="text-green-400 text-xl font-bold">{result}</p>
    </div>
  );
}

export default PredictionResult;
