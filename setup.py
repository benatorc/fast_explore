from setuptools import setup, find_packages

setup(
    name="fast_explore",
    version="0.1.0",
    keywords="chemistry",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "torch>=1.3.0"
    ]
)
