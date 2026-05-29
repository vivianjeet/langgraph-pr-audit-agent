# LangGraph PR Audit Agent 🏦🔒

A multi-agent stateful AI system designed to automate Pull Request (PR) security and quality audits, built for comprehensive security assessments across all domains.

## 📖 What is this project?
In modern software development, every code change requires strict security reviews before merging to prevent vulnerabilities and data leaks. 

This project uses **LangGraph** to orchestrate a team of specialized AI agents that review GitHub PR diffs. It utilizes the **ReAct** (Reason + Act) pattern to deeply analyze code changes against strict security standards (OWASP Top 10, SQL Injection, PII data leaks, Authentication bypasses).

### Core Technologies:
- **LangGraph:** Stateful multi-agent orchestration and routing.
- **Gemini 2.5 Turbo:** Core LLM reasoning engine (via `google-genai`).
- **Instructor:** Enforces strict structured JSON outputs (Pydantic V2 schemas).
- **Python 3.12+:** Core language.

---

## 🏗️ Architecture (In Progress)

Currently, the agent workflow follows this graph topology:
1. **Ingest Node:** Parses raw PR diffs via regex to extract `[ADDED]` and `[REMOVED]` lines securely.
2. **Context Retrieval:** (Stub) Will pull historically similar PRs.
3. **Plan Node:** (Stub) Generates an execution plan.
4. **Audit Nodes (Parallel):**
   - `security_audit_node`: Validates against security vulnerabilities.
   - `quality_audit_node`: (Stub) Validates clean code practices.
   - `test_audit_node`: (Stub) Checks test coverage.
5. **Synthesize & Reflexion:** Routes findings, self-critiques, and formats reports.
6. **Human-in-the-loop:** Pauses execution for human approval before finalization.

---

## 🚀 How to Install & Start

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone <your-repo-link>
cd langgraph-pr-audit-agent

# Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.template` file to `.env` file in the root directory and add your API keys:
```bash
# bash and powershell
cp .env.template .env

# windows command prompt (cmd)
copy .env.template .env
```

---

## 🧪 How to Test

### Run the Unit Tests (Pytest)
Unit tests run instantly and cost $0, asserting that your deterministic logic (like diff parsing) works perfectly.
```bash
# Run tests with verbose output
pytest -v
```

### Run the E2E Smoke Test
The smoke test pushes a sample SQL-injection PR diff through the entire LangGraph state machine and makes live calls to Gemini.
```bash
# Run the full graph smoke test
python main.py --test
```
