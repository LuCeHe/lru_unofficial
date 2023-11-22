import os

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    reqs = fh.read().splitlines()

setup(
    name='lruun',
    version='0.0.2',
    author='Luca Herrtti',
    author_email='luca.herrtti@gmail.com',
    description='LRU TensorFlow implementation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lucehe/lru_unofficial',
    project_urls={
        "Bug Tracker": "https://github.com/lucehe/lru_unofficial/issues"
    },
    license='MIT',
    python_requires=">=3.9",
    packages=find_packages(where="src", exclude=("tests")),
    package_dir={"": "src"},
    install_requires=reqs,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
