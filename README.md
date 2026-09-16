# 🏥 Agentic Clinical Deterioration & Escalation Copilot

A real-time agentic AI system that monitors patient vitals, detects multi-parameter deterioration trends, and escalates the right cases to clinicians with evidence-grounded explanations reducing alarm fatigue in hospital wards.

**Hackathon Track:** Agentic AI Systems (Healthcare)

---

## Problem Description

Hospitals continuously monitor patients using bedside devices, but current alarm systems:

- Fire on **single-threshold breaches** (e.g., HR > 120), producing false alarms
- Cannot detect **multi-parameter trends** (e.g., HR up + RR up + SpO2 down = early sepsis)
- Cause **alarm fatigue**, burying genuinely critical patients under noise

Clinicians watching several patients at once need a system that thinks in **trajectories and combinations**, not isolated readings.

---

## Our Solution

An agentic AI copilot that:

1. **Ingests** a live stream of vitals (HR, SpO2, RR, BP) per patient
2. **Maintains** an evolving stateful profile for each patient
3. **Detects** meaningful multi-parameter deterioration trends (not single readings)
4. **Scores & ranks** patients by urgency using clinical early-warning rules (NEWS2)
5. **Suppresses** redundant alerts for ongoing events
6. **Retrieves** patient context + clinical guidelines to ground its reasoning
7. **Escalates** with a plain-language, evidence-based explanation
8. **Keeps the clinician in the loop** (Accept / Dismiss / Defer / Investigate)
9. **Logs** every observation, decision, and action for full auditability

---

## Architecture

A visual architecture diagram will be added at `docs/architecture.png`.

---

## One-Time Setup 

Do this **once** on your machine. After this, jump to [Daily Workflow].

### 1. Install Prerequisites

| Tool | Link |
|---|---|
| Git | https://git-scm.com/downloads |
| Python 3.11+ | https://www.python.org/downloads/ |
| VS Code (recommended) | https://code.visualstudio.com/ |

Verify installation:

```bash
git --version
python --version
```

### 2. Configure Git (one-time, per machine)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Use the **same email** as your GitHub account.

### 3. Clone the Repository

```bash
git clone https://github.com/neeharika2802/clinical-deterioration-copilot.git
cd clinical-deterioration-copilot
```

### 4. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it based on your OS:

| OS / Shell | Command |
|---|---|
| Windows (Git Bash) | `source .venv/Scripts/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

This file will populate as the project develops.

---

## Daily Workflow (For Team)


```bash
git pull

# Do your work 

# Check what changed
git status

# Stage everything
git add .

# Commit with a meaningful message
git commit -m "Add NEWS2 scoring function"

# Push to GitHub
git push
```

---

## 👥 Team & Task Split

| Member | Responsibility |
|---|---|
| [@neeharika2802](https://github.com/neeharika2802) | -- |
| [@BJ008Creative](https://github.com/BJ008Creative) | -- |


---

## 📁 Project Structure (Planned)

```text
clinical-deterioration-copilot/
├── simulator/           # Vitals stream simulator
├── state/               # Stateful patient profile management
├── agent/               # Agentic reasoning core
│   ├── trend_detector.py
│   ├── risk_scorer.py
│   ├── retrieval.py
│   ├── escalation.py
│   └── explainer.py
├── ui/                  # Clinician-facing interface
├── audit/               # Audit trail logging
├── data/                # Static patient profiles + guidelines
│   ├── patients.json
│   └── news2_rules.json
├── docs/                # Documentation, midterm report
├── tests/               # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Demo

Demo video link will be added here.

### Planned Demo Scenario

1. Simulate 5 patients with evolving vitals
2. Trigger a multi-parameter deterioration event in one patient
3. Show the agent:
   - Detecting the trend
   - Scoring & ranking the cohort
   - Retrieving patient context + NEWS2 guidelines
   - Escalating with an explanation
4. Show the clinician accepting the alert
5. Show the audit log recording the event
6. Show suppression of redundant alerts for the same ongoing event

---

## References

- [NEWS2 National Early Warning Score (Royal College of Physicians)](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2)
- Problem Statement: Agentic Clinical Deterioration & Escalation Copilot

---

## 📄 License

This project is licensed under the MIT License.