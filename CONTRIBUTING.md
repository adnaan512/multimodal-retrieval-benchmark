# Contributing to Multimodal Retrieval Benchmark

Thank you for considering a contribution to this project! This guide explains how to set up a development environment, run the test suite, and submit changes.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Running Tests](#running-tests)
3. [Code Style](#code-style)
4. [Project Structure](#project-structure)
5. [Submitting Changes](#submitting-changes)
6. [Reporting Issues](#reporting-issues)

---

## Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/multimodal-retrieval-benchmark.git
cd multimodal-retrieval-benchmark

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
# or separately:
pip install -r requirements.txt -r requirements-dev.txt
```

---

## Running Tests

The test suite uses **pytest** and runs entirely in mock mode — no CLIP weights or Flickr30K data are required.

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_clip_encoder.py -v

# Run the offline end-to-end demo
python examples/run_demo.py
```

> **CI policy:** The GitHub Actions workflow installs only `numpy`, `pytest`, and `flake8` — no `torch` or `transformers`. Any new feature must remain exercisable through `MockCLIPEncoder` in CI.

---

## Code Style

This project uses **black** for formatting, **isort** for import ordering, and **flake8** for lint.

```bash
# Auto-format
black src/ tests/ main.py examples/ --line-length 130
isort src/ tests/ main.py examples/

# Lint check (mirrors CI)
flake8 src/ main.py tests/ examples/ --max-line-length=130 --extend-ignore=E203,W503

# Optional static type check
mypy src/ --ignore-missing-imports
```

### Docstring Convention

Use **Google-style docstrings** for all public functions and classes:

```python
def recall_at_k(results: List[RetrievalResult], k: int) -> float:
    """Compute Recall@K across all queries.

    Args:
        results: List of retrieval results, one per query.
        k: Cutoff rank. A query is a hit if the ground truth
           appears in the top-K retrieved candidates.

    Returns:
        Fraction of queries that are hits at rank K, in [0, 1].
    """
```

---

## Project Structure

New code should respect the existing module boundaries:

| Module | Responsibility |
|--------|---------------|
| `src/models.py` | Core dataclasses — shared contract between all modules |
| `src/backbone/` | Encoder wrappers only — no retrieval or evaluation logic |
| `src/indexing/` | Embedding caching and retrieval primitives |
| `src/retrieval/` | Direction-specific search and reranking |
| `src/analysis/` | Query categorization and failure analysis |
| `src/benchmark/` | Metric aggregation and comparison |
| `src/reporting/` | HTML report generation |

---

## Submitting Changes

1. **Open an issue first** for any non-trivial change to discuss the approach.
2. Branch from `main`: `git checkout -b feat/your-feature`.
3. Write or update tests for any changed behaviour.
4. Ensure `pytest tests/` passes and `flake8` produces no errors.
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open a pull request with a clear description of what changed and why.

### Commit Message Format

```
<type>(<scope>): <short description>

<optional body>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`

Examples:
```
feat(reranking): add configurable penalty weight via CLI flag
fix(dataset_loader): handle captions.csv with trailing whitespace
docs(RESEARCH): add statistical significance note for 1K split
```

---

## Reporting Issues

Please open a [GitHub issue](https://github.com/adnaan512/multimodal-retrieval-benchmark/issues) and include:

- Python version and OS
- Exact command you ran
- Full error traceback
- Whether the issue reproduces in `--mode demo` (offline, no downloads)
