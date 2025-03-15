<div align="center">

# :robot: Yet Another GPT :robot:
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/python/cpython)
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)

</div>

> [!NOTE]
> TThis repository was created for educational purposes based on [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

---

## :package: Installation
This project requires python >= 3.11.

### Create a new venv
> [!NOTE]
> This will create a new virtual environment in the working directory under .venv.
```bash
# If you have uv installed
uv venv

# Otherwise
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # if you are on Linux
.\.venv\Scripts\activate.bat  # if you are using the cmd on Windows
.\.venv\Scripts\Activate.ps1  # if you are using the PowerShell on Windows
```

### Install the package
> [!NOTE]
> This will install <img height="15" width="15" src="https://cdn.simpleicons.org/pytorch"/>PyTorch compiled with CUDA.
```bash
# Using uv, no dev dependencies
uv pip install git+https://github.com/RistoAle97/yet-another-gpt

# Using uv, installing dev dependencies as well
uv pip install git+https://github.com/RistoAle97/yet-another-gpt[dev]

# Using pip, no dev dependencies
pip install git+https://github.com/RistoAle97/yet-another-gpt

# Using pip, installing dev dependencies as well
pip install "yetanothergpt[dev] @ git+https://github.com/RistoAle97/yet-another-gpt"
```

---

## :memo: License
This project is [MIT licensed](https://github.com/RistoAle97/centered-kernel-alignment/blob/main/LICENSE).
