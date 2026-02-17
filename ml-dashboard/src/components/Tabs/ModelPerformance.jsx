import { MetricsGrid } from '../Metrics';

function ModelPerformance({ metrics }) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-white mb-4">Model Performance Metrics</h2>
      <MetricsGrid metrics={metrics} />
    </div>
  );
}

export default ModelPerformance;
