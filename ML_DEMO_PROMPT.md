# ML Model Development Demo Prompt

Use this prompt to demonstrate the efficient use of Claude Code agents and skills for building a machine learning classification model.

---

## The Prompt

```
Build a classification model to predict Titanic passenger survival using the dataset in data/titanic.csv.

Follow this workflow using the available skills and agents:

1. **Start with /eda** - Perform exploratory data analysis on the Titanic dataset. Understand the features, identify missing values, analyze distributions, and examine correlations with the target variable (Survived).

2. **Use /preprocess** - Create a robust preprocessing pipeline that handles:
   - Missing values in Age, Cabin, and Embarked columns
   - Categorical encoding for Sex, Embarked, and Pclass
   - Numerical scaling for Age and Fare
   - Ensure NO data leakage by using sklearn Pipelines

3. **Use /train** - Train a classification model:
   - Split data 80/20 with stratification
   - Start with LogisticRegression as baseline
   - Try RandomForestClassifier for comparison
   - Use 5-fold cross-validation
   - Apply proper hyperparameter tuning

4. **Use /evaluate** - Perform comprehensive evaluation:
   - Generate confusion matrix
   - Calculate precision, recall, F1, and ROC-AUC
   - Plot ROC curve and feature importance
   - Analyze misclassified samples

Throughout the workflow, the following agents will be automatically engaged:
- **eda-analyst**: Comprehensive data profiling and quality assessment
- **ml-theory-advisor**: Reviews each step for data leakage, overfitting risks, and methodology issues
- **feature-engineering-analyst**: Identifies opportunities for better feature engineering
- **brutal-code-reviewer**: Ensures code quality, naming conventions, and maintainability
- **mlops-engineer**: Guides productionalization, containerization, and deployment

Create all code in a Jupyter notebook at notebooks/titanic_classification.ipynb

5. **Run and validate the notebook** - Execute all cells and verify:
   - All imports work correctly
   - No runtime errors
   - Model achieves reasonable performance (ROC-AUC > 0.80)
   - Visualizations render properly

6. **Productionalize the model** - Extract notebook code into production-ready modules:
   - Create `src/preprocessing.py` with the preprocessing pipeline
   - Create `src/model.py` with model training and prediction functions
   - Create `src/predict.py` as a CLI tool for making predictions
   - Save the trained model to `models/titanic_model.joblib`
   - Add proper error handling, logging, and type hints
   - Use the **brutal-code-reviewer** agent to ensure production code quality

7. **Create tests** - Add unit tests in `tests/`:
   - Test preprocessing handles edge cases (missing values, unknown categories)
   - Test model loading and prediction
   - Test input validation

8. **Document the API** - Create usage documentation:
   - How to train a new model
   - How to make predictions
   - Required dependencies (requirements.txt)

9. **Build an interactive dashboard** - Create a Streamlit app for visualization:
   - Create `app/dashboard.py` with interactive UI
   - Display EDA visualizations (distributions, correlations)
   - Allow users to input passenger data and get predictions
   - Show model performance metrics and feature importance
   - Add data exploration filters and controls
   - Use the **frontend-ux-analyst** agent to review UI/UX

10. **(Optional) Containerize and deploy** - Use the **mlops-engineer** agent to:
   - Create a Dockerfile with proper multi-stage build
   - Create docker-compose.yml for local deployment
   - Add a FastAPI endpoint for serving predictions
   - Set up health checks and logging
   - Create monitoring dashboards (Prometheus/Grafana)
```

---

## Available Resources

### Skills (User-Invocable Commands)
| Skill | Command | Purpose |
|-------|---------|---------|
| EDA | `/eda` | Exploratory data analysis |
| Preprocess | `/preprocess` | Data preprocessing pipeline |
| Train | `/train` | Model training with best practices |
| Evaluate | `/evaluate` | Comprehensive model evaluation |

### Agents (Automatically Invoked)
| Agent | Trigger | Purpose |
|-------|---------|---------|
| `ml-theory-advisor` | ML code/decisions | Prevents data leakage, overfitting |
| `feature-engineering-analyst` | Feature work | Identifies opportunities, anti-patterns |
| `brutal-code-reviewer` | After code written | Ensures code quality and clarity |
| `frontend-ux-analyst` | UI/visualization work | Design and UX improvements |
| `eda-analyst` | Data exploration | Comprehensive data profiling and quality assessment |
| `mlops-engineer` | Deployment/production | Containerization, APIs, CI/CD, monitoring |

---

## Project Structure

```
claude-code-test/
├── .claude/
│   ├── agents/
│   │   ├── ml-theory-advisor.md
│   │   ├── feature-engineering-analyst.md
│   │   ├── brutal-code-reviewer.md
│   │   ├── frontend-ux-analyst.md
│   │   ├── eda-analyst.md
│   │   └── mlops-engineer.md
│   └── skills/
│       ├── eda.md
│       ├── preprocess.md
│       ├── train.md
│       └── evaluate.md
├── data/
│   └── titanic.csv          # 891 samples, 12 features
├── models/
│   └── titanic_model.joblib # Serialized trained model
├── notebooks/
│   └── titanic_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Feature engineering & preprocessing pipeline
│   ├── model.py             # Model training & loading functions
│   └── predict.py           # CLI prediction tool
├── app/
│   └── dashboard.py         # Streamlit interactive dashboard
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_model.py
├── requirements.txt         # Python dependencies
└── ML_DEMO_PROMPT.md        # This file
```

---

## Expected Workflow Demonstration

```
User: [Pastes the prompt above]

Claude: I'll build a Titanic survival classification model using the available skills and agents.

=== PHASE 1: NOTEBOOK DEVELOPMENT ===

[Invokes /eda skill]
→ eda-analyst performs comprehensive data profiling
→ ml-theory-advisor reviews EDA findings for modeling concerns

[Invokes /preprocess skill]
→ ml-theory-advisor validates no data leakage in pipeline
→ feature-engineering-analyst identifies opportunities

[Invokes /train skill]
→ ml-theory-advisor reviews training methodology

[Invokes /evaluate skill]
→ ml-theory-advisor validates evaluation methodology
→ feature-engineering-analyst suggests improvements

=== PHASE 2: EXECUTION & VALIDATION ===

[Runs notebook cells]
→ Verifies all code executes without errors
→ Confirms model performance metrics
→ Validates visualizations render correctly

=== PHASE 3: PRODUCTIONALIZATION ===

[Creates src/preprocessing.py]
→ mlops-engineer guides production code structure
→ Extracts feature engineering logic
→ Creates reusable preprocessing pipeline class

[Creates src/model.py]
→ Implements train() and predict() functions
→ Adds model serialization (save/load)

[Creates src/predict.py]
→ mlops-engineer designs CLI interface
→ CLI tool for batch predictions
→ Input validation and error handling

[Saves model to models/titanic_model.joblib]
→ brutal-code-reviewer reviews all production code
→ mlops-engineer validates serialization approach

[Optional: Creates Dockerfile & docker-compose.yml]
→ mlops-engineer provides containerization templates
→ Creates health check endpoints
→ Sets up proper logging

=== PHASE 4: INTERACTIVE DASHBOARD ===

[Creates app/dashboard.py]
→ Streamlit app with multiple pages/tabs
→ EDA visualizations (interactive charts)
→ Prediction interface (user inputs passenger data)
→ Model insights (feature importance, metrics)

[Dashboard review]
→ frontend-ux-analyst reviews layout and usability
→ eda-analyst validates visualization choices

=== PHASE 5: TESTING & DOCUMENTATION ===

[Creates tests/]
→ Unit tests for preprocessing edge cases
→ Integration tests for full prediction pipeline

[Creates requirements.txt]
→ Lists all dependencies with versions

[Final review]
→ brutal-code-reviewer ensures production readiness
→ ml-theory-advisor validates model serialization preserves pipeline integrity
→ mlops-engineer reviews deployment checklist and monitoring setup
```

---

## Dataset Overview

**Titanic Dataset** (891 passengers)

| Feature | Type | Description |
|---------|------|-------------|
| PassengerId | int | Unique identifier |
| Survived | int | **Target** (0=No, 1=Yes) |
| Pclass | int | Ticket class (1, 2, 3) |
| Name | string | Passenger name |
| Sex | string | Gender |
| Age | float | Age (has missing values) |
| SibSp | int | Siblings/spouses aboard |
| Parch | int | Parents/children aboard |
| Ticket | string | Ticket number |
| Fare | float | Passenger fare |
| Cabin | string | Cabin number (many missing) |
| Embarked | string | Port of embarkation (C/Q/S) |

---

## Tips for Best Results

1. **Let skills guide the workflow** - Each skill has built-in best practices
2. **Trust agent feedback** - Agents catch subtle issues humans miss
3. **Iterate based on findings** - EDA insights should inform preprocessing decisions
4. **Document decisions** - Use markdown cells to explain choices
5. **Compare models** - Always establish a baseline before trying complex models

---

## Productionalization Guidelines

### Model Serialization
```python
# Save the ENTIRE pipeline (preprocessing + model) together
import joblib
joblib.dump(trained_pipeline, 'models/titanic_model.joblib')

# Load for predictions
pipeline = joblib.load('models/titanic_model.joblib')
predictions = pipeline.predict(new_data)
```

### Production Code Structure

**src/preprocessing.py** should contain:
- Feature engineering functions (extract_title, create_family_features)
- Column definitions (NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
- Pipeline creation function

**src/model.py** should contain:
- `train_model(data_path, model_output_path)` - Full training pipeline
- `load_model(model_path)` - Load serialized model
- `predict(model, input_data)` - Make predictions with validation

**src/predict.py** should contain:
- CLI interface using argparse
- Input validation (check required columns exist)
- Output formatting (CSV, JSON)
- Error handling with informative messages

### Example CLI Usage
```bash
# Train a new model
python -m src.model --train --data data/titanic.csv --output models/titanic_model.joblib

# Make predictions
python -m src.predict --model models/titanic_model.joblib --input new_passengers.csv --output predictions.csv

# Single prediction
python -m src.predict --model models/titanic_model.joblib --passenger '{"Pclass": 1, "Sex": "female", "Age": 25}'
```

### Testing Checklist
- [ ] Preprocessing handles missing values correctly
- [ ] Preprocessing handles unknown categories (new titles, embarked ports)
- [ ] Model loads without errors
- [ ] Predictions match expected format
- [ ] Edge cases: empty input, single row, malformed data
- [ ] Performance: prediction latency < 100ms for single sample

### Dependencies (requirements.txt)
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
fastapi>=0.100.0
uvicorn>=0.23.0
streamlit>=1.28.0
plotly>=5.17.0
```

---

## Interactive Dashboard (Streamlit)

The `frontend-ux-analyst` agent will help design an intuitive, user-friendly dashboard.

### Dashboard Structure (app/dashboard.py)

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["🔍 Data Explorer", "📊 Model Insights", "🎯 Make Prediction"])

if page == "🔍 Data Explorer":
    st.header("Exploratory Data Analysis")

    df = pd.read_csv("data/titanic.csv")

    # Interactive filters
    col1, col2 = st.columns(2)
    with col1:
        pclass_filter = st.multiselect("Passenger Class", [1, 2, 3], default=[1, 2, 3])
    with col2:
        sex_filter = st.multiselect("Sex", ["male", "female"], default=["male", "female"])

    filtered_df = df[(df['Pclass'].isin(pclass_filter)) & (df['Sex'].isin(sex_filter))]

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df, x="Age", color="Survived", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(filtered_df, x="Pclass", y="Fare", color="Survived")
        st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader("Raw Data")
    st.dataframe(filtered_df, use_container_width=True)

elif page == "📊 Model Insights":
    st.header("Model Performance & Feature Importance")

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "0.82")
    col2.metric("Precision", "0.79")
    col3.metric("Recall", "0.74")
    col4.metric("ROC-AUC", "0.86")

    # Feature importance chart
    importance_df = pd.DataFrame({
        'Feature': ['Sex', 'Pclass', 'Fare', 'Age', 'Title', 'FamilySize'],
        'Importance': [0.28, 0.18, 0.15, 0.12, 0.10, 0.08]
    })
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Make Prediction":
    st.header("Predict Passenger Survival")

    # Load model
    model = joblib.load("models/titanic_model.joblib")

    # Input form
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 30)
    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare ($)", 0.0, 500.0, 32.0)

    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

    if st.button("Predict Survival", type="primary"):
        # Prepare input
        input_data = pd.DataFrame([{
            'Pclass': pclass, 'Sex': sex, 'Age': age,
            'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # Display result
        if prediction == 1:
            st.success(f"✅ **Survived** (Probability: {probability[1]:.1%})")
        else:
            st.error(f"❌ **Did Not Survive** (Probability: {probability[0]:.1%})")

        # Show probability gauge
        st.progress(probability[1])
```

### Running the Dashboard

```bash
# Install Streamlit
pip install streamlit plotly

# Run the dashboard
streamlit run app/dashboard.py

# Access at http://localhost:8501
```

### Dashboard Pages

| Page | Features |
|------|----------|
| **Data Explorer** | Interactive filters, distribution charts, correlation heatmap, raw data table |
| **Model Insights** | Performance metrics, feature importance, confusion matrix, ROC curve |
| **Make Prediction** | User input form, real-time prediction, probability visualization |

### UX Guidelines (from frontend-ux-analyst)

1. **Clear visual hierarchy** - Most important info at top
2. **Consistent color coding** - Green=survived, Red=died throughout
3. **Responsive layout** - Use columns that adapt to screen size
4. **Immediate feedback** - Show loading spinners, success/error states
5. **Accessible** - Proper contrast, keyboard navigation, alt text
6. **Progressive disclosure** - Advanced options in expandable sections

---

## Containerization & Deployment (Optional)

The `mlops-engineer` agent can help create production-ready deployment artifacts:

### Dockerfile (API)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile.dashboard (Streamlit)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
    restart: unless-stopped
```

### API Endpoints (FastAPI)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model version and metadata |

### Deployment Commands
```bash
# Build and run all services
docker-compose up --build

# Run only the dashboard
docker-compose up dashboard

# Run only the API
docker-compose up ml-api

# Access points:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 1, "Sex": "female", "Age": 25, "Fare": 100}'
```

### Monitoring Checklist
- [ ] Logging configured (predictions, errors, latency)
- [ ] Health check endpoint available
- [ ] Prometheus metrics exposed (optional)
- [ ] Alerting for error rate spikes
- [ ] Model drift monitoring planned
