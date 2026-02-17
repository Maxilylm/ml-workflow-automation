import { useState } from 'react';
import { Header, Footer, Sidebar } from './components/Layout';
import { TabContainer, ModelPerformance, DataExplorer } from './components/Tabs';
import { mockMetrics, simulatePrediction } from './data/mockData';

function App() {
  const [activeTab, setActiveTab] = useState('Model Performance');
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState(null);

  const handlePredict = (f1, f2, f3) => {
    try {
      setPredictionError(null);
      const result = simulatePrediction(f1, f2, f3);
      setPredictionResult(result);
    } catch (err) {
      setPredictionError('Prediction failed. Please try again.');
      setPredictionResult(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <Header />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          onPredict={handlePredict}
          predictionResult={predictionResult}
          predictionError={predictionError}
        />

        <main className="flex-1 p-6 overflow-auto">
          <TabContainer activeTab={activeTab} onTabChange={setActiveTab}>
            {activeTab === 'Model Performance' && (
              <ModelPerformance metrics={mockMetrics} />
            )}
            {activeTab === 'Explore Data' && (
              <DataExplorer />
            )}
          </TabContainer>
        </main>
      </div>

      <Footer />
    </div>
  );
}

export default App;
