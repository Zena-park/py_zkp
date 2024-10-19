#!/usr/bin/env python
from setuptools import (
    find_packages,
    setup,
)

extras_require = {
    "dev": [
        "build>=1.2.2",
    ]
}

with open("./README.md") as readme:
    long_description = readme.read()


setup(
    name="py-zkp",
    version="0.1.0",
    description="py-zkp: ZKP in python including groth16, plonk, tokamak_snarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tokamak Network",
    author_email="kevin@tokamak.network",
    url="https://github.com/tokamak-network/py_zkp",
    include_package_data=True,
    install_requires=["py-ecc==7.0.1"],
    python_requires=">=3.8, <4",
    extras_require=extras_require,
    py_modules=["py_zkp"],
    license="",
    zip_safe=False,
    keywords="zkp",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
