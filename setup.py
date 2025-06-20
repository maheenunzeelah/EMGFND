from setuptools import find_packages, setup

setup(
    name="data_science_project",
    version="0.1.0",
    description="A data science project template",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)