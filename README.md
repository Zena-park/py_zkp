# py_zkp

This is complementaly project of [ZKP study series postings(WIP)](https://www.notion.so/tokamak/6e59b0e13af24a83ae50a10cd59dfbfa?pvs=4)

This is a library that allows you to verify Python code with various ZKP algorithms, including groth16 and plonk.
Using this library, you can convert Python code to QAP and use functions for the entire ZKP process, including setup, proving, and verifying.

# Quickstart

```
python -m pip install py_zkp
```

# Developer Setup

### Development Environment Setup

You can set up your dev environment with:

```
git clone https://github.com/tokamak-network/py_zkp.git
cd py_zkp
python3 -m venv venv
. venv/bin/activate
python -m pip install -e ".[dev]"
```
