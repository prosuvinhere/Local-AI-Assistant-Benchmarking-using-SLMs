# Contributing to the Local SLM Assistant & Benchmarking Suite

Thank you for your interest in contributing! This project is dedicated to reproducible, privacy-first local AI performance engineering.

---

## Code of Conduct
Please be polite, respectful, and constructive in all discussions, issues, and pull requests.

---

## Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/slm-benchmark-suite.git
   cd slm-benchmark-suite
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies in Editable Mode**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Verify Offline Unit Tests**:
   ```bash
   pytest -v
   ```

---

## Architectural Guidelines

- **Offline Independence**: All core logic, schemas, and metrics calculation tests in `tests/` MUST be runnable offline without an active Ollama daemon or downloaded models.
- **Strict Typing & Pydantic Config**: When introducing new output schemas in `app/output_schemas.py`, always specify `model_config = ConfigDict(extra="forbid")` to ensure `additionalProperties: false` propagates to Ollama's constrained grammar decoder.
- **Dual-Clock Telemetry Integrity**: Do not alter `app/metrics.py` timing derivations without updating both the client-side wall-clock formulas and server-side duration parsing logic.
- **Small-Sample Statistics**: Maintain Student-t confidence intervals rather than standard normal distributions when aggregating benchmark runs with $n < 30$.

---

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Ensure all tests pass:
   ```bash
   pytest
   ```
4. Push to your fork and submit a Pull Request targeting `main`.
