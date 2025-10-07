from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy>=1.20.0",
    "torch>=2.0.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}

setup(
    name="bayesian-uq",
    version="0.1.0",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
)
