# Environment setup

The reference platform is Linux x86-64 with Python 3.10 and an NVIDIA GPU for training. The inspected experiment environment used Python 3.10.21, PyTorch 2.1.1+cu118, NumPy 1.23.5, SciPy 1.10.1 and pandas 2.0.3. Its package inventory is in [server_environment.json](../metadata/server_environment.json). This is a current environment snapshot, not an immutable record of every historical run.

## Separate environment for each model

From the repository root, substitute the model name in both paths:

```bash
python3.10 -m venv .venv-gopt
source .venv-gopt/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r environments/gopt.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

| Model | Requirements | Notes |
|---|---|---|
| GOPT | `environments/gopt.txt` | Real-checkpoint CPU forward/backward check passed. |
| HiPAMA | `environments/hipama.txt` | Shared PyTorch stack plus Kaldi readers. |
| HierTFR | `environments/hiertfr.txt` | ESPnet 202402 is retained from upstream instructions; model construction needs ESPnet. |
| ConPCO | `environments/conpco.txt` | Contrastive regularizer is included in the source. |
| HMamba | `environments/hmamba.txt` | Linux/Python 3.10 wheels tied to CUDA 11.8, torch 2.1 and the matching C++ ABI. |
| M3C | `environments/m3c.txt` | Custom attention and pooling layers are included. |
| Analysis only | `environments/analysis.txt` | Does not require PyTorch or a GPU. |

The recipes are derived from the observed server and archived model requirements. They have not all been installed from scratch on a clean machine. The six original training CLIs successfully displayed `--help` in the current server environment; that check does not exercise model construction or CUDA kernels. GOPT has the additional real-checkpoint execution check described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## CUDA and HMamba

The requirements select the cu118 PyTorch build used on the server. PyTorch documents matching `torch==2.1.1`, `torchvision==0.16.1` and `torchaudio==2.1.1` binaries in its [previous-version installation guide](https://pytorch.org/get-started/previous-versions/). The GPU driver must support the selected CUDA runtime. The observed server had an NVIDIA L40S with driver 595.71.05.

HMamba's two compiled wheels are the exact URLs recorded in the server's requirements file. Do not substitute a wheel built for a different Python, PyTorch, CUDA or ABI combination. If a wheel is unavailable, follow the upstream build instructions in [the original HMamba README](../05_hmamba/README.md). Kaldi is additionally needed for the original MDD transcript-scoring pipeline; it is not needed for evaluation of already saved regression predictions.

## Recorded execution pattern

On the experiment server, training was launched from each model's `src/` directory, except HMamba, which runs from its model root. Commands typically used `nohup bash <launcher> > <log> &`; HierTFR supplied `1e-3 1 1 1 1` as launcher arguments. The publication helper records those settings in [launch_configs.json](../metadata/launch_configs.json) and directs each new run to an explicitly chosen directory.

Do not copy the archived virtual environment as an installation mechanism. Virtual environments contain platform-specific executables and absolute paths. This repository distributes dependency recipes and the observed package inventory instead.
