<div align="center">

# :robot: Yet Another GPT :robot:
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/python/cpython)
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)

</div>

> [!NOTE]
> This repository was created for educational purposes based on [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

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

## :hammer_and_wrench: Architecture overview

<div>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/RistoAle97/yet-another-gpt/blob/main/assets/gpt_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/RistoAle97/yet-another-gpt/blob/main/assets/gpt_light.png">
    <img align="right" alt="GPT architecture" src="https://github.com/RistoAle97/yet-another-gpt/blob/main/assets/gpt_dark.png" height=390, width=250>
  </picture>
</div>

  ```python
  import torch
  from yetanothergpt import GPTConfig, GPT


  # Some configurations from the original implementation
  small_config = GPTConfig(n_layers=12, n_heads=12, d_model=768)  # 124M params
  medium_config = GPTConfig(n_layers=24, n_heads=16, d_model=1024)  # 350M params
  large_config = GPTConfig(n_layers=36, n_heads=20, d_model=1280)  # 774M params
  xl_config = GPTConfig(n_layers=48, n_heads=25, d_model=1600)  # 1.558B params

  # Set up the model
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = GPT(small_config).to(device)

  # Try a forward pass
  input_tokens = torch.randint(0, model.config.vocab_size, size=(10, 128)).to(device)
  pad_mask = torch.zeros_like(input_tokens, dtype=torch.bool).to(device)
  logits, loss = model(input_tokens, pad_mask)
  ```

---

## :memo: License
This project is [MIT licensed](https://github.com/RistoAle97/centered-kernel-alignment/blob/main/LICENSE).
