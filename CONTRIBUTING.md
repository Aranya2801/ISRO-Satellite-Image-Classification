# 🤝 Contributing to ISRO Satellite Image Classification

Thank you for your interest in contributing! This project welcomes contributions of all kinds — bug fixes, new features, documentation improvements, and more.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

---

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this standard. Please report unacceptable behavior to the maintainers.

---

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ISRO-Satellite-Image-Classification.git
   cd ISRO-Satellite-Image-Classification
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/Aranya2801/ISRO-Satellite-Image-Classification.git
   ```

---

## 🛠️ Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
python scripts/verify_install.py
```

---

## ✏️ Making Changes

1. **Sync** with upstream:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a branch** using Conventional Commits format:
   ```bash
   git checkout -b feat/swin-v2-backbone     # New feature
   git checkout -b fix/ndvi-computation-bug  # Bug fix
   git checkout -b docs/api-reference        # Documentation
   git checkout -b refactor/dataset-loader   # Refactoring
   ```

3. **Make your changes** following the [Coding Standards](#coding-standards)

4. **Test** your changes:
   ```bash
   pytest tests/ -v --cov=src
   ```

5. **Commit** using [Conventional Commits](https://conventionalcommits.org/):
   ```bash
   git commit -m "feat(models): add Swin-V2 backbone with relative position bias"
   git commit -m "fix(preprocessing): correct NDVI computation for negative values"
   git commit -m "docs(readme): add UCM dataset download instructions"
   ```

---

## 🔀 Pull Request Process

1. **Push** your branch:
   ```bash
   git push origin feat/your-feature-name
   ```

2. Open a **Pull Request** against `main` with:
   - Clear title following Conventional Commits
   - Description of what changed and why
   - Screenshots / benchmark results (if applicable)
   - Reference to related issues (`Closes #123`)

3. Ensure all **CI checks pass**:
   - ✅ Lint (black, isort, flake8)
   - ✅ Tests (pytest, coverage ≥ 80%)
   - ✅ Docker build
   - ✅ Security scan

4. Request review from `@Aranya2801`

---

## 📐 Coding Standards

### Python Style
- **Formatter**: [Black](https://black.readthedocs.io/) (line length: 100)
- **Import sorting**: [isort](https://pycqa.github.io/isort/)
- **Linter**: [flake8](https://flake8.pycqa.org/) with `--max-line-length=100`
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public classes and functions

### Example Function
```python
def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)
    Range: [-1, 1] | Dense vegetation > 0.4

    Args:
        nir: Near-infrared band array (H, W), values in [0, 1].
        red: Red band array (H, W), values in [0, 1].
        eps: Small constant to avoid division by zero.

    Returns:
        NDVI array (H, W) with values in [-1, 1].

    Example:
        >>> ndvi = compute_ndvi(nir_band, red_band)
        >>> print(f"Mean NDVI: {ndvi.mean():.3f}")
    """
    return (nir - red) / (nir + red + eps)
```

### File Organization
- **Models**: One class per file in `src/models/`
- **Tests**: Mirror `src/` structure in `tests/`
- **Configs**: YAML files only in `configs/`

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Specific module
pytest tests/unit/test_spectral_indices.py -v

# Fast (skip slow tests)
pytest tests/ -v -m "not slow"
```

### Writing Tests
```python
# tests/unit/test_spectral_indices.py
import numpy as np
import pytest
from src.preprocessing.spectral_indices import compute_ndvi

class TestNDVI:
    def test_basic_computation(self):
        nir = np.array([[0.8, 0.6], [0.4, 0.9]])
        red = np.array([[0.2, 0.3], [0.5, 0.1]])
        ndvi = compute_ndvi(nir, red)
        assert ndvi.shape == (2, 2)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <=  1.0

    def test_vegetation_positive(self):
        """High NIR, low Red → positive NDVI (vegetation)."""
        nir = np.ones((3, 3)) * 0.8
        red = np.ones((3, 3)) * 0.1
        ndvi = compute_ndvi(nir, red)
        assert (ndvi > 0).all()
```

---

## 🆕 Adding a New Model

1. Create `src/models/your_model.py` following the pattern in `resnet_satellite.py`
2. Register it in `MODEL_REGISTRY` in `src/models/ensemble.py`
3. Add config block in `configs/config.yaml`
4. Write unit tests in `tests/unit/test_models.py`
5. Update the benchmark table in `README.md`

---

## 📦 Adding a New Dataset

1. Add dataset info to `DATASETS` dict in `scripts/download_datasets.py`
2. Create a new `Dataset` class in `src/preprocessing/dataset.py`
3. Register class names in `DATASET_CLASSES` dict
4. Update `configs/config.yaml` with dataset-specific stats
5. Add EDA in a new notebook

---

Thank you for contributing! 🛰️ Every PR, issue, and suggestion makes this project better.
