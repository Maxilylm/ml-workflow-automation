# ML Dashboard React Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the Streamlit ML Model Dashboard to a React application with Tailwind CSS styling and mock data.

**Architecture:** Component-per-section organization with Layout, Metrics, Prediction, and Tabs component groups. State managed in App.jsx, mock data in dedicated data module.

**Tech Stack:** Create React App, React 18, Tailwind CSS 3, PostCSS

---

## Task 1: Project Scaffolding

**Files:**
- Create: `ml-dashboard/` (via CRA)
- Modify: `ml-dashboard/package.json`
- Modify: `ml-dashboard/src/index.css`
- Create: `ml-dashboard/tailwind.config.js`
- Create: `ml-dashboard/postcss.config.js`

**Step 1: Create React App**

```bash
cd /Users/maximolorenzoylosada/Documents/claude-code-test
npx create-react-app ml-dashboard
```

Expected: New `ml-dashboard/` directory with CRA boilerplate

**Step 2: Install Tailwind dependencies**

```bash
cd /Users/maximolorenzoylosada/Documents/claude-code-test/ml-dashboard
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Expected: `tailwind.config.js` and `postcss.config.js` created

**Step 3: Configure Tailwind**

Replace `ml-dashboard/tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Step 4: Add Tailwind directives**

Replace `ml-dashboard/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 5: Verify setup works**

```bash
cd /Users/maximolorenzoylosada/Documents/claude-code-test/ml-dashboard
npm start
```

Expected: App runs at localhost:3000 without errors

**Step 6: Commit**

```bash
cd /Users/maximolorenzoylosada/Documents/claude-code-test
git add ml-dashboard/
git commit -m "feat: scaffold React app with Tailwind CSS"
```

---

## Task 2: Mock Data Module

**Files:**
- Create: `ml-dashboard/src/data/mockData.js`

**Step 1: Create data directory and mock data file**

Create `ml-dashboard/src/data/mockData.js`:

```javascript
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
```

**Step 2: Commit**

```bash
git add ml-dashboard/src/data/
git commit -m "feat: add mock data module for metrics and predictions"
```

---

## Task 3: Layout Components

**Files:**
- Create: `ml-dashboard/src/components/Layout/Header.jsx`
- Create: `ml-dashboard/src/components/Layout/Footer.jsx`
- Create: `ml-dashboard/src/components/Layout/index.js`

**Step 1: Create Layout directory structure**

```bash
mkdir -p ml-dashboard/src/components/Layout
```

**Step 2: Create Header component**

Create `ml-dashboard/src/components/Layout/Header.jsx`:

```jsx
function Header() {
  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center gap-3">
        <span className="text-2xl">🤖</span>
        <h1 className="text-xl font-semibold text-white">ML Model Dashboard</h1>
      </div>
    </header>
  );
}

export default Header;
```

**Step 3: Create Footer component**

Create `ml-dashboard/src/components/Layout/Footer.jsx`:

```jsx
function Footer() {
  return (
    <footer className="bg-gray-800 border-t border-gray-700 px-6 py-3 text-center">
      <p className="text-gray-400 text-sm">
        Built with React | ML Dashboard
      </p>
    </footer>
  );
}

export default Footer;
```

**Step 4: Create index barrel export**

Create `ml-dashboard/src/components/Layout/index.js`:

```javascript
export { default as Header } from './Header';
export { default as Footer } from './Footer';
```

**Step 5: Commit**

```bash
git add ml-dashboard/src/components/Layout/
git commit -m "feat: add Header and Footer layout components"
```

---

## Task 4: MetricCard and MetricsGrid Components

**Files:**
- Create: `ml-dashboard/src/components/Metrics/MetricCard.jsx`
- Create: `ml-dashboard/src/components/Metrics/MetricsGrid.jsx`
- Create: `ml-dashboard/src/components/Metrics/index.js`

**Step 1: Create Metrics directory**

```bash
mkdir -p ml-dashboard/src/components/Metrics
```

**Step 2: Create MetricCard component**

Create `ml-dashboard/src/components/Metrics/MetricCard.jsx`:

```jsx
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
```

**Step 3: Create MetricsGrid component**

Create `ml-dashboard/src/components/Metrics/MetricsGrid.jsx`:

```jsx
import MetricCard from './MetricCard';

function MetricsGrid({ metrics }) {
  const metricItems = [
    { label: 'Accuracy', value: metrics.accuracy },
    { label: 'Precision', value: metrics.precision },
    { label: 'Recall', value: metrics.recall },
    { label: 'F1 Score', value: metrics.f1 },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {metricItems.map((item) => (
        <MetricCard key={item.label} label={item.label} value={item.value} />
      ))}
    </div>
  );
}

export default MetricsGrid;
```

**Step 4: Create index barrel export**

Create `ml-dashboard/src/components/Metrics/index.js`:

```javascript
export { default as MetricCard } from './MetricCard';
export { default as MetricsGrid } from './MetricsGrid';
```

**Step 5: Commit**

```bash
git add ml-dashboard/src/components/Metrics/
git commit -m "feat: add MetricCard and MetricsGrid components"
```

---

## Task 5: Prediction Components

**Files:**
- Create: `ml-dashboard/src/components/Prediction/PredictionForm.jsx`
- Create: `ml-dashboard/src/components/Prediction/PredictionResult.jsx`
- Create: `ml-dashboard/src/components/Prediction/index.js`

**Step 1: Create Prediction directory**

```bash
mkdir -p ml-dashboard/src/components/Prediction
```

**Step 2: Create PredictionForm component**

Create `ml-dashboard/src/components/Prediction/PredictionForm.jsx`:

```jsx
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
```

**Step 3: Create PredictionResult component**

Create `ml-dashboard/src/components/Prediction/PredictionResult.jsx`:

```jsx
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
```

**Step 4: Create index barrel export**

Create `ml-dashboard/src/components/Prediction/index.js`:

```javascript
export { default as PredictionForm } from './PredictionForm';
export { default as PredictionResult } from './PredictionResult';
```

**Step 5: Commit**

```bash
git add ml-dashboard/src/components/Prediction/
git commit -m "feat: add PredictionForm and PredictionResult components"
```

---

## Task 6: Tab Components

**Files:**
- Create: `ml-dashboard/src/components/Tabs/TabContainer.jsx`
- Create: `ml-dashboard/src/components/Tabs/ModelPerformance.jsx`
- Create: `ml-dashboard/src/components/Tabs/DataExplorer.jsx`
- Create: `ml-dashboard/src/components/Tabs/index.js`

**Step 1: Create Tabs directory**

```bash
mkdir -p ml-dashboard/src/components/Tabs
```

**Step 2: Create TabContainer component**

Create `ml-dashboard/src/components/Tabs/TabContainer.jsx`:

```jsx
function TabContainer({ activeTab, onTabChange, children }) {
  const tabs = ['Model Performance', 'Explore Data'];

  return (
    <div>
      <div className="flex border-b border-gray-700 mb-4">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => onTabChange(tab)}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === tab
                ? 'text-white border-b-2 border-blue-500'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>
      {children}
    </div>
  );
}

export default TabContainer;
```

**Step 3: Create ModelPerformance component**

Create `ml-dashboard/src/components/Tabs/ModelPerformance.jsx`:

```jsx
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
```

**Step 4: Create DataExplorer component**

Create `ml-dashboard/src/components/Tabs/DataExplorer.jsx`:

```jsx
function DataExplorer() {
  return (
    <div>
      <h2 className="text-lg font-semibold text-white mb-4">Data Explorer</h2>
      <div className="bg-blue-900/30 border border-blue-700 rounded p-4">
        <p className="text-blue-300">
          Add your data exploration queries here
        </p>
      </div>
    </div>
  );
}

export default DataExplorer;
```

**Step 5: Create index barrel export**

Create `ml-dashboard/src/components/Tabs/index.js`:

```javascript
export { default as TabContainer } from './TabContainer';
export { default as ModelPerformance } from './ModelPerformance';
export { default as DataExplorer } from './DataExplorer';
```

**Step 6: Commit**

```bash
git add ml-dashboard/src/components/Tabs/
git commit -m "feat: add TabContainer, ModelPerformance, and DataExplorer components"
```

---

## Task 7: Sidebar Component

**Files:**
- Modify: `ml-dashboard/src/components/Layout/Sidebar.jsx`
- Modify: `ml-dashboard/src/components/Layout/index.js`

**Step 1: Create Sidebar component**

Create `ml-dashboard/src/components/Layout/Sidebar.jsx`:

```jsx
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
```

**Step 2: Update Layout index export**

Replace `ml-dashboard/src/components/Layout/index.js`:

```javascript
export { default as Header } from './Header';
export { default as Footer } from './Footer';
export { default as Sidebar } from './Sidebar';
```

**Step 3: Commit**

```bash
git add ml-dashboard/src/components/Layout/
git commit -m "feat: add Sidebar component with prediction form"
```

---

## Task 8: Main App Integration

**Files:**
- Modify: `ml-dashboard/src/App.jsx`
- Delete: `ml-dashboard/src/App.css`
- Delete: `ml-dashboard/src/App.test.js`
- Delete: `ml-dashboard/src/logo.svg`

**Step 1: Clean up CRA boilerplate**

```bash
rm -f ml-dashboard/src/App.css ml-dashboard/src/App.test.js ml-dashboard/src/logo.svg
```

**Step 2: Create main App component**

Replace `ml-dashboard/src/App.js` (rename to App.jsx):

```bash
mv ml-dashboard/src/App.js ml-dashboard/src/App.jsx
```

Replace content of `ml-dashboard/src/App.jsx`:

```jsx
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
```

**Step 3: Update index.js entry point**

Verify `ml-dashboard/src/index.js` imports correctly (should already work).

**Step 4: Verify app runs**

```bash
cd /Users/maximolorenzoylosada/Documents/claude-code-test/ml-dashboard
npm start
```

Expected: Dashboard renders with sidebar, header, metrics grid, tabs, and footer

**Step 5: Commit**

```bash
git add ml-dashboard/src/
git commit -m "feat: integrate all components into main App"
```

---

## Task 9: Final Verification and Cleanup

**Files:**
- Modify: `ml-dashboard/src/reportWebVitals.js` (optional cleanup)
- Modify: `ml-dashboard/public/index.html`

**Step 1: Update page title**

In `ml-dashboard/public/index.html`, change:
```html
<title>React App</title>
```
to:
```html
<title>ML Model Dashboard</title>
```

**Step 2: Run final verification**

```bash
cd /Users/maximolorenzoylosada/Documents/claude-code-test/ml-dashboard
npm start
```

Verify:
- [ ] Header displays "ML Model Dashboard" with robot icon
- [ ] Sidebar shows 3 inputs and Predict button
- [ ] Clicking Predict shows result below form
- [ ] Two tabs work: "Model Performance" and "Explore Data"
- [ ] Metrics grid shows 4 cards with percentages
- [ ] Footer displays attribution
- [ ] Dark theme throughout

**Step 3: Final commit**

```bash
git add ml-dashboard/
git commit -m "feat: complete ML Dashboard React conversion"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project scaffolding | CRA + Tailwind setup |
| 2 | Mock data module | `data/mockData.js` |
| 3 | Layout components | Header, Footer |
| 4 | Metrics components | MetricCard, MetricsGrid |
| 5 | Prediction components | PredictionForm, PredictionResult |
| 6 | Tab components | TabContainer, ModelPerformance, DataExplorer |
| 7 | Sidebar component | Sidebar |
| 8 | App integration | App.jsx |
| 9 | Final verification | Title, testing |

Total: 9 tasks, ~15-20 minutes estimated
