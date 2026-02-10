# Claude Code ML Workflow

A comprehensive machine learning workflow powered by Claude Code's agents, commands (skills), and hooks system. This setup provides an end-to-end ML pipeline from raw data to deployed models.

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd claude-code-test

# Place your data in data/raw/
cp your_data.csv data/raw/

# Launch the full ML pipeline
/team-coldstart data/raw/your_data.csv --target your_target_column
```

## Directory Structure

```
.
├── .claude/
│   ├── agents/           # Specialized AI agents
│   ├── commands/         # User-invocable skills (slash commands)
│   └── settings.local.json
├── data/
│   └── raw/              # Place your raw datasets here
└── README.md
```

## Agents

Agents are specialized AI personas that handle specific aspects of the ML workflow. They are automatically invoked by Claude Code when appropriate.

| Agent | Purpose | Auto-Triggers When |
|-------|---------|-------------------|
| **ml-theory-advisor** | Prevents data leakage, overfitting, validates methodology | Creating features, training models, evaluating results |
| **eda-analyst** | Comprehensive exploratory data analysis | Analyzing new datasets |
| **feature-engineering-analyst** | Feature design, interaction analysis, opportunity discovery | Developing new features |
| **brutal-code-reviewer** | Code quality, maintainability, AI-friendliness | After significant code changes |
| **mlops-engineer** | Production deployment, containerization, CI/CD | Deploying models |
| **frontend-ux-analyst** | UI/UX design analysis | Creating dashboards |

### How Agents Work

Agents are defined in `.claude/agents/*.md` files. Each agent has:
- **name**: Identifier for the agent
- **description**: When to invoke (with examples)
- **model**: Which Claude model to use (sonnet/opus/haiku)
- **System prompt**: Detailed instructions and expertise

Agents are invoked automatically based on context or explicitly via the Task tool.

## Commands (Skills)

Commands are user-invocable workflows triggered with `/command-name`. They orchestrate complex multi-step processes.

### ML Pipeline Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `/eda` | Exploratory data analysis | `/eda data/raw/dataset.csv` |
| `/preprocess` | Build preprocessing pipeline | `/preprocess --target Survived` |
| `/train` | Train ML models | `/train --model xgboost` |
| `/evaluate` | Comprehensive model evaluation | `/evaluate` |
| `/test` | Generate and run tests | `/test src/preprocessing.py` |
| `/deploy` | Deploy model to various targets | `/deploy local` or `/deploy snowflake` |
| `/report` | Generate ad-hoc reports | `/report performance` |

### Team Orchestration Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `/team-coldstart` | Full ML pipeline from data to deployment | `/team-coldstart data/raw/data.csv` |
| `/team-analyze` | Quick analysis without full pipeline | `/team-analyze data/raw/data.csv` |
| `/team-review` | Multi-agent code review | `/team-review src/` |

### Example: Full Pipeline

```bash
# Start a complete ML project
/team-coldstart data/raw/titanic.csv --target Survived

# This will:
# 1. Create feature branch
# 2. Run EDA (eda-analyst)
# 3. Check for data leakage (ml-theory-advisor)
# 4. Recommend features (feature-engineering-analyst)
# 5. Build preprocessing pipeline
# 6. Train models
# 7. Evaluate (with ml-theory-advisor validation)
# 8. Productionalize (mlops-engineer)
# 9. Deploy (optional)
# 10. Generate final report
```

## Hooks

Hooks are shell commands that execute in response to Claude Code events. Configure in `.claude/settings.local.json` or `~/.claude/settings.json`.

### Hook Events

| Event | When It Fires |
|-------|---------------|
| `PreToolUse` | Before any tool executes |
| `PostToolUse` | After any tool completes |
| `Notification` | When Claude sends a notification |
| `Stop` | When Claude stops responding |

### Example Hook Configuration

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "echo 'About to run: $TOOL_INPUT'"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "command": "python scripts/lint_check.py $FILE_PATH"
      }
    ]
  }
}
```

## Permissions

The `.claude/settings.local.json` file configures allowed operations:

```json
{
  "permissions": {
    "allow": [
      "Bash(python3:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "WebFetch(domain:www.anthropic.com)"
    ]
  }
}
```

## Workflow Examples

### 1. Quick Data Analysis

```bash
# Just want to understand your data?
/eda data/raw/customers.csv

# EDA analyst will:
# - Profile all columns
# - Check data quality
# - Analyze distributions
# - Find correlations
# - Identify issues
```

### 2. Model Training with Safety

```bash
# Train a model with automatic leakage checks
/train data/raw/sales.csv --target revenue

# ml-theory-advisor automatically:
# - Reviews feature engineering for leakage
# - Validates train/test split
# - Checks evaluation methodology
```

### 3. Deploy to Production

```bash
# Deploy locally with Docker
/deploy local

# Deploy to Snowflake
/deploy snowflake

# Deploy to cloud
/deploy aws --region us-east-1
```

### 4. Code Review

```bash
# After writing code, get a thorough review
/team-review src/

# brutal-code-reviewer will check:
# - Naming conventions
# - Code clarity
# - Documentation quality
# - Maintainability
```

## Agent Coordination

The workflow coordinates multiple agents:

```
User Request
     │
     ▼
┌─────────────────┐
│  team-coldstart │ (orchestrator)
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
┌───────┐ ┌──────────┐ ┌─────────────┐
│  eda  │ │ml-theory │ │feature-eng. │
│analyst│ │ advisor  │ │  analyst    │
└───┬───┘ └────┬─────┘ └──────┬──────┘
    │          │              │
    └──────────┴──────────────┘
               │
               ▼
        (preprocessing)
               │
               ▼
        ┌──────────────┐
        │ ml-theory    │ (validation)
        │ advisor      │
        └──────┬───────┘
               │
               ▼
          (training)
               │
               ▼
        ┌──────────────┐
        │   mlops-     │
        │  engineer    │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │brutal-code-  │
        │  reviewer    │
        └──────────────┘
```

## Creating Custom Agents

Add new agents in `.claude/agents/`:

```markdown
---
name: my-custom-agent
description: "Description of when to use this agent"
model: sonnet
color: blue
---

System prompt defining the agent's expertise and behavior...
```

## Creating Custom Commands

Add new commands in `.claude/commands/`:

```markdown
---
name: my-command
description: "What this command does"
user_invocable: true
---

Instructions for what Claude should do when this command is invoked...
```

## Best Practices

1. **Always start with EDA** - Understand your data before modeling
2. **Let ml-theory-advisor review** - It catches leakage automatically
3. **Use team-coldstart for new projects** - Full pipeline with all checks
4. **Run tests before deployment** - `/test` enforces 80% coverage
5. **Review generated code** - brutal-code-reviewer ensures quality

## Requirements

- Claude Code CLI
- Python 3.11+
- Docker (for local deployment)
- Git

## License

MIT
