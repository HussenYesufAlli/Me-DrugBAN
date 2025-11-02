from setuptools import setup, find_packages

setup(
    name="me-drugban",
    version="0.0.1",
    description="Reimplementation of DrugBAN",
    author="Hussen Yesuf Alli",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)