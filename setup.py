"""Package configuration for ModelForge Local."""

from setuptools import find_packages, setup

setup(
    name="modelforge-local",
    version="0.1.0",
    description="Local-first AutoML and active learning platform",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "Bootstrap-Flask==2.5.0",
        "Flask==3.1.1",
        "Flask-Migrate==4.1.0",
        "Flask-SQLAlchemy==3.1.1",
        "python-dotenv==1.1.1",
        "Pillow==11.3.0",
        "numpy==2.3.2",
        "pandas==2.3.1",
        "scikit-learn==1.7.1",
        "joblib==1.5.1",
    ],
)
