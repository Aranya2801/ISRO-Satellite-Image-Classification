"""
ISRO Satellite Image Classification — Package Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="isro-satellite-classification",
    version="2.0.0",
    author="Aranya2801",
    author_email="",
    description="Advanced deep learning ensemble for ISRO satellite image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aranya2801/ISRO-Satellite-Image-Classification",
    project_urls={
        "Bug Tracker": "https://github.com/Aranya2801/ISRO-Satellite-Image-Classification/issues",
        "Documentation": "https://github.com/Aranya2801/ISRO-Satellite-Image-Classification/blob/main/docs/",
        "Source Code": "https://github.com/Aranya2801/ISRO-Satellite-Image-Classification",
    },
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "pytest-asyncio>=0.23",
            "black>=24.0",
            "isort>=5.13",
            "flake8>=7.0",
            "mypy>=1.10",
            "pre-commit>=3.7",
        ],
        "gpu": [
            "torch>=2.2.0+cu118",
            "torchvision>=0.17.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "isro-train=src.train:main",
            "isro-download=scripts.download_datasets:main",
            "isro-api=src.deployment.api:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "satellite imagery", "remote sensing", "deep learning",
        "land cover classification", "ISRO", "EuroSAT",
        "image classification", "PyTorch", "ensemble",
        "ResNet", "Vision Transformer", "EfficientNet",
    ],
    include_package_data=True,
    zip_safe=False,
)
