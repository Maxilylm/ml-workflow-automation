# Claude Code ML Workflow — Presentation & Demo Resources

> **This repository contains presentation slides and demo resources only.**
> For the actual ML automation workflow, use the **ML Automation Plugin** instead.

## Use the Plugin

The workflow demonstrated here is available as a ready-to-install plugin for Claude Code, Cursor, Codex, and OpenCode:

**[https://github.com/maxilylm/ml-automation-plugin](https://github.com/maxilylm/ml-automation-plugin)** (v1.5.3)

### Install in Claude Code (recommended)

```bash
# Option 1: From marketplace
# In Claude Code, run /plugin → Add marketplace → maxilylm/ml-automation-plugin

# Option 2: Local clone
git clone https://github.com/maxilylm/ml-automation-plugin.git
claude --plugin-dir /path/to/ml-automation-plugin
```

### Install in other editors

| Platform | Command |
|----------|---------|
| **Cursor** | Clone and place where Cursor discovers plugins |
| **Codex** | `git clone https://github.com/maxilylm/ml-automation-plugin.git ~/.codex/ml-automation` |
| **OpenCode** | `git clone https://github.com/maxilylm/ml-automation-plugin.git ~/.config/opencode/ml-automation` |

See the [plugin README](https://github.com/maxilylm/ml-automation-plugin) for full installation instructions per platform.

## What the Plugin Provides

### 10 Specialized Agents

| Agent | Role |
|-------|------|
| `eda-analyst` | Exploratory data analysis on any dataset |
| `ml-theory-advisor` | ML theory guidance, data leakage prevention |
| `feature-engineering-analyst` | Feature design and opportunity discovery |
| `mlops-engineer` | Model deployment, APIs, Docker, CI/CD |
| `developer` | Code implementation on feature branches |
| `brutal-code-reviewer` | Code quality and maintainability review |
| `pr-approver` | Pull request review and merge |
| `frontend-ux-analyst` | UI/UX design feedback |
| `orchestrator` | Multi-agent coordination |
| `assigner` | Automatic ticket routing |

### 12 Skills / Slash Commands

| Skill | Description |
|-------|-------------|
| `/eda` | Run exploratory data analysis |
| `/preprocess` | Build data processing pipeline (leakage-safe) |
| `/train` | Train ML models with proper validation |
| `/evaluate` | Comprehensive model evaluation with visualizations |
| `/deploy` | Deploy to Docker, Snowflake, AWS, or GCP |
| `/report` | Generate EDA, model, drift, or project reports |
| `/test` | Generate and run tests (80% coverage threshold) |
| `/team-coldstart` | Full pipeline: raw data to deployed dashboard |
| `/team-analyze` | Quick multi-agent data analysis |
| `/team-review` | Multi-agent code review |
| `/status` | Show workflow status and agent reports |
| `/registry` | Inspect MLOps registries (models, features, experiments, data) |

### Hooks

- **Pre-commit**: Python syntax check, secrets detection, test coverage validation
- **Pre-deploy**: Deployment readiness checks
- **Post-EDA**: Extract metrics, flag data quality issues
- **Post-dashboard**: Validate syntax, detect placeholders, import-level check
- **Post-workflow**: Summarize outputs, generate quick-start commands

## Workflow Overview

```
Raw Data → /eda → /preprocess → /train → /evaluate → /deploy
              ↓         ↓           ↓          ↓
         eda-analyst  ml-theory  ml-theory  mlops-engineer
                      advisor    advisor
```

Or run the full pipeline with a single command:

```bash
/team-coldstart data/sales.csv --target Revenue
```

This orchestrates all stages automatically:
1. Validates data and detects task type
2. Runs parallel EDA, leakage review, and feature analysis
3. Builds preprocessing pipeline with leakage prevention
4. Trains and compares models
5. Generates comprehensive evaluation
6. Creates interactive Streamlit dashboard
7. Packages for production (FastAPI + Docker)
8. Deploys to target environment

### Key Features (v1.5.3)

- **Evaluation Framework** — 30 evals with 78 assertions covering all skills and agent routing
- **Self-Check Loops** — Every stage validates its output before proceeding
- **Pre-Stage Reflection** — Domain expert agents plan the approach before each stage
- **Lessons Learned** — Persistent knowledge base across workflow runs
- **Reflection Gates** — Pre-execution validation of strategy before each stage
- **MLOps Registry** — Track models, features, experiments, and data versions
- **Shared Report Bus** — Cross-agent communication through JSON reports
- **Parallel Agent Execution** — Concurrent agent runs where dependencies allow

## Presentation

Open `presentation.html` in a browser to view the slide deck covering Claude Code extensibility concepts and the ML automation workflow.

## Requirements

- Claude Code CLI (or Cursor / Codex / OpenCode)
- Python 3.9+
- Common ML libraries: pandas, scikit-learn, matplotlib, seaborn
- Optional: streamlit, fastapi, docker

## License

MIT
