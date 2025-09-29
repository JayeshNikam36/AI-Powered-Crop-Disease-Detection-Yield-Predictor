from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="AI Powered Crop Disease Detection",
    version="0.1.0",
    author="Jayesh",
    packages=find_packages(),
    install_requires=requirements,
    description="Real-time crop disease detection using drones + AI, with yield forecasting.",
)