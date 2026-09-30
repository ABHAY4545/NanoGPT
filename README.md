# NanoGPT

This is a tutorial follow-along of [Andrej Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt) and his [Let's reproduce GPT-2 (124M) video](https://www.youtube.com/watch?v=l8pRSuU81PU). The code closely follows his example; I made this project to learn the basics of GPT-2 training and inference.

## Setup

Requires Python 3.12 or newer. Training requires a CUDA GPU and tokenized FineWeb-Edu data.

```bash
uv sync
```

## Training

Create the tokenized data shards, then start training:

```bash
uv run python scripts/fineweb.py
uv run python NanoGPT.py
```

The data script downloads the FineWeb-Edu 10B-token sample into `data/edu_fineweb10B/`.

## Inference

`inference.py` loads a local checkpoint from `checkpoints/parameter_checkpoint_19072.pt`. Checkpoints are not included in this repository. Place the checkpoint there, or change `checkpoint_path` in the script, then run:

```bash
uv run python inference.py
```
