from setuptools import setup, find_packages

setup(
    name="gym_retro",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1,<1.1.0",
        "numpy"
    ],
)
