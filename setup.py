"""
setup.py — Optional packaging configuration for the KYC Verification System.

Install in development mode:
    pip install -e .

Build a distributable wheel:
    pip install build
    python -m build
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "models" / "README.md").read_text(encoding="utf-8")

setup(
    name="kyc-verification-system",
    version="1.0.0",
    description="AI-powered Indian Multi-Document KYC & Facial Verification Desktop Application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="KYC System Author",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    install_requires=[
        "customtkinter==5.2.2",
        "CTkMessagebox==2.5",
        "Pillow==10.1.0",
        "opencv-python==4.8.1.78",
        "paddlepaddle==2.5.2",
        "paddleocr==2.7.3",
        "insightface==0.7.3",
        "onnxruntime==1.16.3",
        "torch==2.1.2",
        "torchvision==0.16.2",
        "timm==0.9.7",
        "xgboost==2.0.3",
        "scikit-learn==1.3.2",
        "numpy==1.26.2",
        "scipy==1.11.4",
    ],
    entry_points={
        "console_scripts": [
            "kyc-system=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
