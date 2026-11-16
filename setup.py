from setuptools import find_packages, setup


setup(
    name="driver-drowsiness-system",
    version="0.1.0",
    description="A camera-based driver drowsiness monitoring prototype",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)