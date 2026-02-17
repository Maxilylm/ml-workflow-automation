# ML Dashboard React Conversion Design

**Date:** 2026-02-17
**Status:** Approved

## Overview

Convert the Streamlit ML Model Dashboard (`deploy/snowflake/streamlit/app.py`) to a React application using Create React App and Tailwind CSS.

## Decisions

| Decision | Choice |
|----------|--------|
| Framework | Create React App |
| Styling | Tailwind CSS |
| Data Source | Mock data (static) |
| Architecture | Component-Per-Section |

## Project Structure

```
ml-dashboard/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Layout/
│   │   │   ├── Header.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── Footer.jsx
│   │   ├── Metrics/
│   │   │   ├── MetricCard.jsx
│   │   │   └── MetricsGrid.jsx
│   │   ├── Prediction/
│   │   │   ├── PredictionForm.jsx
│   │   │   └── PredictionResult.jsx
│   │   └── Tabs/
│   │       ├── TabContainer.jsx
│   │       ├── ModelPerformance.jsx
│   │       └── DataExplorer.jsx
│   ├── data/
│   │   └── mockData.js
│   ├── App.jsx
│   ├── index.js
│   └── index.css
├── tailwind.config.js
├── package.json
└── README.md
```

## Components

### Layout Components

- **Header**: Dashboard title with icon, dark background
- **Sidebar**: Fixed left sidebar with prediction input form
- **Footer**: Attribution text

### Metrics Components

- **MetricCard**: Single metric display (label, value as percentage)
- **MetricsGrid**: 4-column responsive grid containing MetricCards

### Prediction Components

- **PredictionForm**: Three inputs (Feature 1, Feature 2, Feature 3 dropdown) + Predict button
- **PredictionResult**: Displays prediction result with success (green) or error (red) styling

### Tab Components

- **TabContainer**: Manages active tab state, renders tab buttons
- **ModelPerformance**: Contains MetricsGrid, shows model metrics
- **DataExplorer**: Placeholder tab with info message

## Data Flow

1. Mock data in `data/mockData.js` provides static metrics
2. `App.jsx` holds state for active tab and prediction result
3. `Sidebar` contains `PredictionForm` which calls prediction handler
4. Prediction simulates API response with computed value based on inputs
5. State updates trigger re-render of `PredictionResult`

## Mock Data Structure

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
  return (f1 * 0.3 + f2 * 0.5 + f3 * 10).toFixed(2);
};
```

## Styling Approach

- Dark theme matching Streamlit appearance
- Tailwind utility classes for all styling
- Key colors:
  - Background: `bg-gray-900` (primary), `bg-gray-800` (secondary)
  - Text: `text-white` (primary), `text-gray-400` (secondary)
  - Accent: `text-green-400` (success), `text-red-400` (error)
- Responsive breakpoints for metrics grid

## Dependencies

```json
{
  "dependencies": {
    "react": "^18.x",
    "react-dom": "^18.x"
  },
  "devDependencies": {
    "tailwindcss": "^3.x",
    "autoprefixer": "^10.x",
    "postcss": "^8.x"
  }
}
```

## Out of Scope

- Real API integration (mock data only)
- Authentication
- Data persistence
- Unit tests (manual testing sufficient)
- Charts/visualizations in Data Explorer tab
