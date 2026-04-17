"""
setup.py
========
Install the text_to_image_gan package in development mode:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="text_to_image_gan",
    version="1.0.0",
    description="Text-to-Image GAN Pipeline on 102 Oxford Flowers",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "nltk>=3.8.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "tensorboard>=2.13.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "eval": ["pytorch-fid>=0.3.0"],
        "dev":  ["pytest", "black", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "t2i-train    = train:main",
            "t2i-generate = generate:main",
            "t2i-evaluate = evaluate:main",
        ],
    },
)
