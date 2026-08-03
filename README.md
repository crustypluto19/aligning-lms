<div align="center">

# Aligned DP-BART

Preference-aligned differentially private text privatization.

</div>

Anonymous repository for the August 2026 ARR submission: *Aligning Language Models for Differentially Private Text Generation:
A Human-centered Pathway for Better Acceptability and Usability*

## Installation

Install PyTorch for your system first, matching your CUDA or Metal setup. See [pytorch.org](https://pytorch.org/get-started/locally/).

Then install the package:

```bash
pip install -e .
```

If you need the scoring pipeline (BLEURT, BERTScore, etc.), install the optional extras:

```bash
pip install -e ".[scoring]"
pip install git+https://github.com/google-research/bleurt.git
```

BLEURT is not on PyPI, so the separate `git+` install is required.

## Data and models

Training data is included in `data/training/`. Aligned model checkpoints can be downloaded from: [Models](https://drive.google.com/drive/folders/1imx_YY7cBRF_VaTrY8W_FkzuBem5DPiK?usp=sharing)

Place checkpoints in `models/`.

## Quick start

```python
import torch
from transformers import BartTokenizer
from aligned_dp import DPBartAligned, PrivatizationPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DPBartAligned.from_baseline("facebook/bart-base", epsilon=1500)
model = model.to(device).eval()

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
pipeline = PrivatizationPipeline(model, tokenizer)

out = pipeline("The food at this restaurant was absolutely incredible.")
print(out)
```

## CLI tools

The package has three command-line tools. Run any of them with `--help` for full usage.

| Command | What it does |
|---------|-------------|
| `python -m aligned_dp.train` | Train the decoder on preference data (DPO by default, `--use-vpo` for VPO) |
| `python -m aligned_dp.privatize` | Generate privatized datasets across multiple epsilons |
| `python -m aligned_dp.score` | Compute Metric Acceptability scores (utility, structure, readability) |

## Human evaluation survey

See [`survey/README.md`](survey/README.md) for setup and usage.

<details>
<summary><strong>Repository structure</strong></summary>

```
aligning-lms/
├── src/
│   └── aligned_dp/               # Core Python package
│       ├── __init__.py
│       ├── dp_bart_aligned.py     # DPBartAligned model (encoder hook + DP noise)
│       ├── dp_bart_pr_plus.py     # PR+ baseline wrapper
│       ├── dp_utils.py            # DP utility functions
│       ├── vpo_trainer.py         # VPO trainer with synchronized noise
│       ├── metric_acceptability.py # Metric Acceptability scoring engine
│       ├── train.py               # Training CLI
│       ├── privatize.py           # Generation CLI
│       └── score.py               # Scoring CLI
├── survey/                        # Streamlit human evaluation app
├── data/                          # Training datasets
├── pyproject.toml
└── LICENSE
```

</details>

## License

[MIT](LICENSE)
