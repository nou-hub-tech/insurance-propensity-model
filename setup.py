from setuptools import setup, find_packages

setup(
    name="insurance-mlops",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "dvc==3.51.2",
        "dvc-gdrive==3.0.1",
        "fastapi==0.111.1",
        "imbalanced-learn==0.12.3",
        "joblib==1.4.2",
        "mlflow==2.14.3",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "PyYAML==6.0.1",
        "scikit-learn==1.5.1",
        "uvicorn==0.30.1",
        "prometheus-client==0.20.0",
        "python-dotenv==1.0.1",
        "evidently==0.4.27",
        "tenacity==8.3.0",
    ],
    python_requires=">=3.8",
)
