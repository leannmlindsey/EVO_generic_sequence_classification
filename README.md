# Evo: DNA foundation modeling from molecular to genome scale

> **Note:** This is a fork of the [original Evo repository](https://github.com/evo-design/evo) with added support for **embedding extraction and analysis**. This allows you to extract embeddings from DNA sequences using the Evo model and perform downstream classification analysis.

---

## Embedding Analysis

This fork adds the ability to extract embeddings from the Evo model and evaluate them using linear probes, neural networks, and visualization.

The embedding extraction approach is based on the solutions discussed in [GitHub Issue #32](https://github.com/evo-design/evo/issues/32) and [GitHub Issue #93](https://github.com/evo-design/evo/issues/93).

### 1. Prepare Your Data

Create a directory containing three CSV files with `sequence` and `label` columns:

```
my_dataset/
├── train.csv
├── dev.csv    # (or val.csv)
└── test.csv
```

Each CSV should have this format:
```csv
sequence,label
ACGTACGTACGT...,0
TGCATGCATGCA...,1
GGCCAATTGGCC...,0
```

- `sequence`: DNA sequence (A, C, G, T characters)
- `label`: Integer class label (0, 1 for binary classification)

### 2. Run Embedding Analysis

```bash
python -m evo.embedding_analysis \
    --csv_dir="/path/to/my_dataset" \
    --model_name="evo-1-8k-base" \
    --output_dir="./results/embedding_analysis" \
    --pooling="mean" \
    --include_random_baseline  # Optional: compare with random embeddings
```

**Key Options:**
- `--model_name`: Evo model to use (`evo-1.5-8k-base`, `evo-1-8k-base`, `evo-1-131k-base`, `evo-1-8k-crispr`, `evo-1-8k-transposon`)
- `--pooling`: Pooling strategy (`mean`, `first`, `last`)
- `--batch_size`: Batch size for embedding extraction (default: 8, reduce if OOM)
- `--max_length`: Maximum sequence length (default: 8192)
- `--layer_idx`: Layer index for intermediate embeddings (default: final layer)
- `--skip_nn`: Only run linear probe, skip neural network training
- `--cache_embeddings`: Cache embeddings to disk for reuse

### 3. SLURM Scripts (for HPC)

SLURM scripts are provided in `slurm_scripts/` for running on HPC clusters:

1. Edit `wrapper_run_embedding_analysis.sh` with your paths
2. Submit: `bash slurm_scripts/wrapper_run_embedding_analysis.sh`
3. For interactive testing: `bash slurm_scripts/wrapper_run_embedding_analysis.sh --interactive`

### 4. Outputs

- `embedding_analysis_results.json`: All metrics (accuracy, precision, recall, F1, MCC, AUC, sensitivity, specificity, silhouette scores)
- `embeddings_pretrained.npz`: Cached embeddings for train/val/test sets
- `pca_visualization_pretrained.png`: PCA plot showing class separation
- `test_predictions_pretrained.csv`: Test predictions with probabilities
- `three_layer_nn_pretrained.pt`: Trained 3-layer neural network model

**Caching:** Embeddings are cached in `.npz` files when `--cache_embeddings` is used. Delete them to re-extract with different settings.

---

## Original Evo README

The remainder of this README is from the [original Evo repository](https://github.com/evo-design/evo).

---

**We have developed a new model called Evo 2 that extends the Evo 1 model and its ideas to all domains of life. Please see [https://github.com/arcinstitute/evo2](https://github.com/arcinstitute/evo2) for more details.**

![Evo](evo.jpg)

Evo is a biological foundation model capable of long-context modeling and design.
Evo uses the [StripedHyena architecture](https://github.com/togethercomputer/stripedhyena) to enable modeling of sequences at a single-nucleotide, byte-level resolution with near-linear scaling of compute and memory relative to context length.
Evo has 7 billion parameters and is trained on [OpenGenome](https://huggingface.co/datasets/LongSafari/open-genome), a prokaryotic whole-genome dataset containing ~300 billion tokens.

We describe Evo in the paper [“Sequence modeling and design from molecular to genome scale with Evo”](https://www.science.org/doi/10.1126/science.ado9336).

We describe Evo 1.5 in the paper [“Semantic design of functional de novo genes from a genomic language model”](https://www.nature.com/articles/s41586-025-09749-7). We used the Evo 1.5 model to generate [SynGenome](https://evodesign.org/syngenome/), the first AI-generated genomics database containing over 100 billion base pairs of synthetic DNA sequences.

We provide the following model checkpoints:
| Checkpoint Name                        | Description |
|----------------------------------------|-------------|
| `evo-1.5-8k-base`   | A model pretrained with 8,192 context obtained by extending the pretraining of `evo-1-8k-base` to process 50% more training data. |
| `evo-1-8k-base`     | A model pretrained with 8,192 context. We use this model as the base model for molecular-scale finetuning tasks. |
| `evo-1-131k-base`   | A model pretrained with 131,072 context using `evo-1-8k-base` as the base model. We use this model to reason about and generate sequences at the genome scale. |
| `evo-1-8k-crispr`   | A model finetuned using `evo-1-8k-base` as the base model to generate CRISPR-Cas systems. |
| `evo-1-8k-transposon`   | A model finetuned using `evo-1-8k-base` as the base model to generate IS200/IS605 transposons. |

## News

**December 17, 2024:** We have found and fixed a bug in the code for Evo model inference affecting package versions from Nov 15-Dec 16, 2024, which has been corrected in release versions 0.3 and above. If you installed the package during this timeframe, please upgrade to correct the issue.

## Contents

- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
- [HuggingFace](#huggingface)
- [Together API](#together-api)
- [colab](https://colab.research.google.com/github/evo-design/evo/blob/main/scripts/hello_evo.ipynb)
- [Playground wrapper](https://evo.nitro.bio/)
- [Dataset](#dataset)
- [Citation](#citation)

## Setup

### Requirements

Evo is based on [StripedHyena](https://github.com/togethercomputer/stripedhyena/tree/main).

Evo uses [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), which may not work on all GPU architectures.
Please consult the [FlashAttention GitHub repository](https://github.com/Dao-AILab/flash-attention#installation-and-features) for the current list of supported GPUs. Currently, Evo supports FlashAttention versions <= 2.7.4.post0.

Make sure to install the correct [PyTorch version](https://pytorch.org/) on your system. PyTorch versions >= 2.7.0 and < 2.8.0a0 are supported by FlashAttention 2.7.4.

We recommend using a fresh conda environment to install these prerequisites. Below is an example of how to install these:
```bash
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge flash-attn=2.7.4
```

### Installation

You can install Evo using `pip`
```bash
pip install evo-model
```
or directly from the GitHub source
```bash
git clone https://github.com/evo-design/evo.git
cd evo/
pip install .
```

If you are not using the conda-forge FlashAttention installation shown above, which will automatically install PyTorch, we recommend that you install the PyTorch library before installing all other dependencies (due to dependency issues of the `flash-attn` library; see, e.g., this [issue](https://github.com/Dao-AILab/flash-attention/issues/246)).

One of our [example scripts](scripts/), demonstrating how to go from generating sequences with Evo to folding proteins ([scripts/generation_to_folding.py](scripts/generation_to_folding.py)), further requires the installation of `prodigal`. We have created an [environment.yml](environment.yml) file for this:

```bash
conda env create -f environment.yml
conda activate evo-design
```

### Troubleshooting

If you are using [Numpy](https://numpy.org/) versions > 2.2, you may encounter the following error:

```bash
ValueError: The binary mode of fromstring is removed, use frombuffer instead
```

To fix this, modify [`tokenizer.py`](https://github.com/togethercomputer/stripedhyena/blob/main/stripedhyena/tokenizer.py#L157) at line 157 in your local installation of [StripedHyena](https://github.com/togethercomputer/stripedhyena) as shown: 

```bash
# Replace this:
return list(np.fromstring(text, dtype=np.uint8))

# With this:
return list(np.frombuffer(text.encode(), dtype=np.uint8))
```

## Usage

Below is an example of how to download Evo and use it locally through the Python API.
```python
from evo import Evo
import torch

device = 'cuda:0'

evo_model = Evo('evo-1-131k-base')
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()

sequence = 'ACGT'
input_ids = torch.tensor(
    tokenizer.tokenize(sequence),
    dtype=torch.int,
).to(device).unsqueeze(0)

with torch.no_grad():
    logits, _ = model(input_ids) # (batch, length, vocab)

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)
```
An example of batched inference can be found in [`scripts/example_inference.py`](scripts/example_inference.py).

We provide an [example script](scripts/generate.py) for how to prompt the model and sample a set of sequences given the prompt.
```bash
python -m scripts.generate \
    --model-name 'evo-1-131k-base' \
    --prompt ACGT \
    --n-samples 10 \
    --n-tokens 100 \
    --temperature 1. \
    --top-k 4 \
    --device cuda:0
```

We also provide an [example script](scripts/score.py) for using the model to score the log-likelihoods of a set of sequences.
```bash
python -m scripts.score \
    --input-fasta examples/example_seqs.fasta \
    --output-tsv scores.tsv \
    --model-name 'evo-1-131k-base' \
    --device cuda:0
```

## HuggingFace

Evo is integrated with [HuggingFace](https://huggingface.co/togethercomputer/evo-1-131k-base).
```python
from transformers import AutoConfig, AutoModelForCausalLM

model_name = 'togethercomputer/evo-1-8k-base'

model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
model_config.use_cache = True

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=model_config,
    trust_remote_code=True,
    revision="1.1_fix"
)
```


## Together API

Evo is available through Together AI with a [web UI](https://api.together.xyz/playground/language/togethercomputer/evo-1-131k-base), where you can generate DNA sequences with a chat-like interface.

For more detailed or batch workflows, you can call the Together API with a simple example below.


```python
import openai
import os

# Fill in your API information here.
client = openai.OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz',
)

chat_completion = client.chat.completions.create(
  messages=[
    {
      "role": "system",
      "content": ""
    },
    {
      "role": "user",
      "content": "ACGT", # Prompt the model with a sequence.
    }
  ],
  model="togethercomputer/evo-1-131k-base",
  max_tokens=128, # Sample some number of new tokens.
  logprobs=True
)
print(
    chat_completion.choices[0].logprobs.token_logprobs,
    chat_completion.choices[0].message.content
)
```

## Dataset

The OpenGenome dataset for pretraining Evo is available at [Hugging Face datasets](https://huggingface.co/datasets/LongSafari/open-genome).

## Citation

Please cite the following publication when referencing Evo.

```
@article{nguyen2024sequence,
   author = {Eric Nguyen and Michael Poli and Matthew G. Durrant and Brian Kang and Dhruva Katrekar and David B. Li and Liam J. Bartie and Armin W. Thomas and Samuel H. King and Garyk Brixi and Jeremy Sullivan and Madelena Y. Ng and Ashley Lewis and Aaron Lou and Stefano Ermon and Stephen A. Baccus and Tina Hernandez-Boussard and Christopher Ré and Patrick D. Hsu and Brian L. Hie },
   title = {Sequence modeling and design from molecular to genome scale with Evo},
   journal = {Science},
   volume = {386},
   number = {6723},
   pages = {eado9336},
   year = {2024},
   doi = {10.1126/science.ado9336},
   URL = {https://www.science.org/doi/abs/10.1126/science.ado9336},
}
```

Please cite the following publication when referencing Evo 1.5.

```
@article{merchant2025semantic,
    author = {Merchant, Aditi T and King, Samuel H and Nguyen, Eric and Hie, Brian L},
    title = {Semantic design of functional de novo genes from a genomic language model},
    year = {2025},
    doi = {10.1038/s41586-025-09749-7},
    URL = {https://www.nature.com/articles/s41586-025-09749-7},
    journal = {Nature}
}
```
