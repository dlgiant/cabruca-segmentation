from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cabruca-segmentation",
    version="1.0.0",
    author="Cabruca Segmentation Team",
    description="ML-based segmentation system for Cabruca agroforestry analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cabruca-segmentation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "albumentations>=1.3.0",
        "pycocotools>=2.0.6",
        "rasterio>=1.3.0",
        "geopandas>=0.12.0",
        "shapely>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.25.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "training": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "mlflow>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cabruca-train=training.train_cabruca:main",
            "cabruca-infer=inference.cabruca_inference:main",
            "cabruca-api=api.inference_api:main",
        ],
    },
)
EOF < /dev/null