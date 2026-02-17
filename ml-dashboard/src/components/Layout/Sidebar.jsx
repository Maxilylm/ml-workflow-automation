import { PredictionForm, PredictionResult } from '../Prediction';

function Sidebar({ onPredict, predictionResult, predictionError }) {
  return (
    <aside className="w-64 bg-gray-800 border-r border-gray-700 p-4 flex-shrink-0">
      <h2 className="text-lg font-semibold text-white mb-4">Model Input</h2>
      <PredictionForm onPredict={onPredict} />
      <PredictionResult result={predictionResult} error={predictionError} />
    </aside>
  );
}

export default Sidebar;
