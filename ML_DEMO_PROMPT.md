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
- **ml-theory-advisor**: Reviews each step for data leakage, overfitting risks, and methodology issues
- **brutal-code-reviewer**: Ensures code quality, naming conventions, and maintainability
- **feature-engineering-analyst**: Identifies opportunities for better feature engineering

Create all code in a Jupyter notebook at notebooks/titanic_classification.ipynb
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

---

## Project Structure

```
claude-code-test/
├── .claude/
│   ├── agents/
│   │   ├── ml-theory-advisor.md
│   │   ├── feature-engineering-analyst.md
│   │   ├── brutal-code-reviewer.md
│   │   └── frontend-ux-analyst.md
│   └── skills/
│       ├── eda.md
│       ├── preprocess.md
│       ├── train.md
│       └── evaluate.md
├── data/
│   └── titanic.csv          # 891 samples, 12 features
├── notebooks/
│   └── (notebooks created here)
├── src/
│   └── (Python modules here)
└── ML_DEMO_PROMPT.md         # This file
```

---

## Expected Workflow Demonstration

```
User: [Pastes the prompt above]

Claude: I'll build a Titanic survival classification model using the available skills and agents.

[Invokes /eda skill]
→ ml-theory-advisor reviews EDA findings for modeling concerns

[Invokes /preprocess skill]
→ ml-theory-advisor validates no data leakage in pipeline

[Invokes /train skill]
→ ml-theory-advisor reviews training methodology
→ brutal-code-reviewer checks code quality

[Invokes /evaluate skill]
→ ml-theory-advisor validates evaluation methodology
→ feature-engineering-analyst suggests improvements
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
