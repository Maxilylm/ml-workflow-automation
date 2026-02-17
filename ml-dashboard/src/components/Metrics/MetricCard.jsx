function MetricCard({ label, value }) {
  const percentage = (value * 100).toFixed(0);

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <p className="text-gray-400 text-sm mb-1">{label}</p>
      <p className="text-2xl font-bold text-white">{percentage}%</p>
    </div>
  );
}

export default MetricCard;
