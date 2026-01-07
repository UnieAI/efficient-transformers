# Repository Guidelines

## Project Structure & Module Organization
- `QEfficient/`: core library (base kernels, transformers/generation, peft, cloud/compile/exporter utilities, custom ops, helpers in `utils/`).
- `tests/`: mirrors the package layout (`transformers/`, `generation/`, `peft/`, `cloud/`, etc.); add new tests beside the code they cover.
- `examples/` and `notebooks/`: reference onboarding guides for new model support and runnable demos.
- `docs/`: Sphinx sources; update when adding flags, models, or APIs.
- `scripts/`: focused workflows (finetune, perplexity, KV replication, specialization) used in CI and benchmarking.

## Build, Test, and Development Commands
- Environment: Python 3.10 recommended. Create a venv and install dev extras:  
  `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[test,quality]`
- Build wheel for distribution: `python -m build --wheel --outdir dist`
- Run lint/format checks: `pre-commit run --all-files` or `ruff check`
- Run the library locally once installed: `python -c "import QEfficient; print(QEfficient.__version__)"` (use `examples/` scripts for end-to-end runs).

## Coding Style & Naming Conventions
- Python 3.8–3.10 targets; use 4-space indentation and descriptive type hints where available.
- Formatting: Black-compatible style, Ruff linting with 120-char line length and isort rules (see `pyproject.toml`).
- Naming: modules/files in `snake_case.py`, classes in `PascalCase`, functions/variables/tests in `snake_case`, tests start with `test_`.
- Keep feature docs/examples in sync (`docs/source/**`, `examples/**`) and prefer reusing existing utilities before adding new ones.

## Testing Guidelines
- Default: `pytest tests` (verbose by configuration). For targeted runs:  
  `pytest tests/transformers/test_causal_lm.py -k llama`
- Hardware markers: skip QAIC-only or long-running suites when not available, e.g., `pytest -m "not on_qaic and not nightly"`.
- New features should include parity checks against reference HF outputs when applicable and cover both PyTorch and exported/ONNX paths.

## Commit & Pull Request Guidelines
- Sign every commit (DCO): `git commit -s -m "Add <feature>: <short detail>"`
- Keep messages concise and descriptive (see `git log` for style); group related changes in one PR.
- PRs should state the motivation, key changes, and test evidence (commands + results). Link related issues/model cards and attach logs or screenshots for user-facing changes.
- Ensure pre-commit/Ruff and relevant pytest suites pass before requesting review; respond to feedback promptly.

## Architecture & Performance Notes
- Typical flow: start from HF model -> apply QEfficient transforms/adapters -> export to ONNX/QNN -> run on Cloud AI 100; verify outputs after each stage.
- Large model assets rely on Hugging Face; cache responsibly and avoid committing weights.
