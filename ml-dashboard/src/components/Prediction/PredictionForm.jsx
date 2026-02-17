import { useState } from 'react';
import { feature3Options } from '../../data/mockData';

function PredictionForm({ onPredict }) {
  const [feature1, setFeature1] = useState(50);
  const [feature2, setFeature2] = useState(50);
  const [feature3, setFeature3] = useState(1);

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(feature1, feature2, feature3);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-gray-300 text-sm mb-1">Feature 1</label>
        <input
          type="number"
          min="0"
          max="100"
          value={feature1}
          onChange={(e) => setFeature1(Number(e.target.value))}
          className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
        />
      </div>

      <div>
        <label className="block text-gray-300 text-sm mb-1">Feature 2</label>
        <input
          type="number"
          min="0"
          max="100"
          value={feature2}
          onChange={(e) => setFeature2(Number(e.target.value))}
          className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
        />
      </div>

      <div>
        <label className="block text-gray-300 text-sm mb-1">Feature 3</label>
        <select
          value={feature3}
          onChange={(e) => setFeature3(Number(e.target.value))}
          className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
        >
          {feature3Options.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      </div>

      <button
        type="submit"
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded transition-colors"
      >
        Predict
      </button>
    </form>
  );
}

export default PredictionForm;
