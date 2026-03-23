from setuptools import setup, find_packages

setup(
    name="quant",
    version="0.1.0",
    description="A Python quantitative trading library",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
)
