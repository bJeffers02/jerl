from setuptools import setup, find_packages

setup(
    name="jerl",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
    'numpy>=1.21.0,<1.27.0',
    ],
    python_requires='>=3.10',
)