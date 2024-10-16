# py-zkp

This is complementaly project of [ZKP study series postings(WIP)](https://www.notion.so/tokamak/6e59b0e13af24a83ae50a10cd59dfbfa?pvs=4)

This is a library that allows you to verify Python code with various ZKP algorithms, including groth16 and plonk.
Using this library, you can convert Python code to QAP and use functions for the entire ZKP process, including setup, proving, and verifying.

# Quickstart

It's not ready yet. It will be published.

```
python -m pip install py-zkp
```

# Developer Setup

### 1. Prerequisite

- python (https://www.python.org/downloads/)
- poetry (https://python-poetry.org/docs/#installation)

### 2. Virtual Environment

```
python -m venv .venv
. .venv/bin/activate
```

### 3. Install Dependencies

```
(.venv) poetry install
```

### 4. Test

```
(.venv) python ./tests/test.py
```

### 5. ETC

Add required package

```
(.venv) poetry add py-zkp
```

Remove required package

```
(.venv) poetry remove py-zkp
```

Build package

```
(.venv) poetry build
```

Publish package

```
(.venv) poetry publish
```
