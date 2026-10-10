# Gist Graph

[![CI](https://github.com/dhrrishitvdeka/gistgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrrishitvdeka/gistgraph/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.10-3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrrishitvdeka/gistgraph/blob/main/notebooks/train_colab.ipynb)

Compress a long text context into a small **latent graph** that a frozen LLM reads as a soft prompt,
aiming to keep about 90% of downstream question-answering performance at 3-5x compression.

> **Status: pipeline complete, first trained model in progress.** Everything from data to a
> published model is implemented and tested: baselines, three compressor families, training,
> evaluation, an inference API and a Colab notebook that trains and releases the model. The
> non-learned baselines have been evaluated on HotpotQA. No compressor has finished training yet,
> so this repository makes **no performance claim** for the method. Trained-model results will be
> added to [docs/results](docs/results/results.md) as they exist, including negative ones.

## Contents

- [How it works](#how-it-works)
- [Hypotheses](#hypotheses)
- [Results so far](#results-so-far)
- [Train on Google Colab](#train-on-google-colab)
- [Install locally](#install-locally)
- [Use a trained model](#use-a-trained-model)
- [Run experiments locally](#run-experiments-locally)
- [Milestones and design notes](#milestones-and-design-notes)
- [Layout](#layout)

## How it works

```
long context ──► segment encoder ──► routed writes into latent nodes ──► sparse typed edges
   (tokens)       (frozen-LLM            (top-k router, gates decide       (bilinear scores +
                   features)              how many nodes stay open)         entmax, no parser)
                                                                                  │
answer ◄── frozen LLM ◄── soft prompt ◄── projector + graph PE ◄── message passing ┘
```

1. A segment encoder turns the context into segment representations, using features from a middle
   layer of the frozen LLM.
2. Segments write into latent nodes through sparse top-k routing, so information-dense text can
   claim more capacity. Hard-concrete gates decide how many nodes stay open, and a rate controller
   holds the open fraction at the target compression ratio.
3. Sparse typed edges between nodes are induced by bilinear scoring and entmax, with no external
   parser. Each node can also choose "no edge".
4. A few message-passing layers encode multi-hop relations in the node states.
5. The nodes are projected into the frozen LLM's embedding space (with a random-walk positional
   encoding of the graph) and read as a soft prompt in front of the question.
6. New text can be merged into existing nodes (streaming memory).

Training distills the frozen LLM's answers given the **full** context into its answers given the
**compressed** context. The teacher's answers and top-64 logits are computed once and cached, so
training never runs the long full-context pass again. A reconstruction loss stabilises early
training and is annealed away. The LLM itself is never updated.

## Hypotheses

- **H1** Graph structure beats flat latent slots at equal node count.
- **H2** Adaptive routing beats a fixed compression ratio.
- **H3** Gains concentrate on multi-hop tasks (HotpotQA, 2WikiMultihopQA) over single-hop (SQuAD).
- **H4** Incremental compression retains quality relative to one-shot compression.

Controls for each (no edges, random graph, fixed rate, flat slots, one-shot) are built in. See
[docs/related_work.md](docs/related_work.md) for prior work and how this project differs.

## Results so far

Non-learned baselines on HotpotQA (200 examples) with `Qwen/Qwen2.5-0.5B-Instruct` as the reader.
The full-context reference scores 30.3 F1 / 18.0 EM; the table shows F1 retained relative to it,
with the compression ratio actually achieved in brackets.

| method | 2x | 3x | 4x | 5x |
|---|---|---|---|---|
| extractive_bm25 (sees the question) | 102% (2.0x) | 92% (3.0x) | 97% (4.0x) | 89% (5.1x) |
| extractive_lead | 88% (2.0x) | 78% (3.0x) | 78% (4.1x) | 76% (5.1x) |
| extractive_textrank | 73% (2.0x) | 69% (3.0x) | 66% (4.0x) | 63% (5.1x) |
| llmlingua2 | 90% (1.9x) | 75% (2.7x) | 55% (3.6x) | - |

These set the bar a learned compressor has to clear. The learned compressors do not see the
question, so the query-aware BM25 baseline is a deliberately strong reference. Full tables,
confidence intervals and how to read them: [docs/results/results.md](docs/results/results.md).

**What the first training attempt found.** The first full run of the graph model showed its edges
collapsing around the end of learning-rate warm-up: an unbounded "no edge" score overtook every
edge, and entmax then gave the edges no gradient to recover, so the graph model had silently become
the routed model. This is fixed (the "no edge" score is now relative to each node's best edge and
bounded) and covered by a regression test; details in [docs/design/m4.md](docs/design/m4.md).
The Colab notebook logs `mean_degree` so a collapse is visible during training.

## Train on Google Colab

The quickest way to train, evaluate and publish a model, with no local GPU:

1. Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Colab (badge above) and pick
   **Runtime → Change runtime type → GPU**. A free T4 is enough; an L4 or A100 is faster.
2. Run the cells in order. The notebook:
   - mounts Google Drive and keeps runs and caches there, so a dropped session resumes from its
     last checkpoint when you re-run the cells;
   - evaluates the baselines on the same examples as the learned models;
   - trains and evaluates the graph (M4) and flat-slot (M2) compressors at 2-5x compression;
   - writes the results page and tries the exported model on an example;
   - publishes the model to the Hugging Face Hub (needs a write token) together with its results.
3. Download `results.zip` from the last cell to bring the results back into the repository.

Precision is chosen automatically: bf16 on GPUs that support it natively (Ampere and newer), fp16
on T4-class GPUs.

## Install locally

```bash
git clone https://github.com/dhrrishitvdeka/gistgraph.git
cd gistgraph
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # use ".[dev,baselines]" to include LLMLingua-2
pytest
ruff check . && ruff format --check .
```

Python 3.10-3.12 and PyTorch 2.6 or newer. The tests use a tiny randomly initialised model and need
no GPU or download. Training or evaluating with the real model needs a CUDA GPU; the default
training configs peak at about 4.3 GB of GPU memory, so an 8 GB card is enough.

## Use a trained model

`GistGraph` wraps a trained compressor and its frozen LLM for inference, with no experiment
harness. The pretrained model `dhrrishitvdeka/gistgraph-qwen2.5-0.5b` will be published on the
Hugging Face Hub once it has been trained and evaluated; until then, use a local run or export.

```python
from gistgraph import GistGraph

gg = GistGraph.from_pretrained("dhrrishitvdeka/gistgraph-qwen2.5-0.5b")  # or a local directory
mem = gg.compress(long_text, ratio=4)        # about a quarter as many nodes as context tokens
print(mem.n_tokens, mem.n_nodes, mem.ratio)
print(gg.answer("Who founded the company?", memory=mem))

# or in one call, and for several questions at once
gg.answer("Who founded the company?", context=long_text, ratio=4)
gg.answer_batch(questions, contexts, ratio=4)
```

A compressed memory can be reused for many questions about the same context. `ratio` must be at
least the smallest ratio the model was trained with (2 for the shipped configs), and contexts longer
than the trained limit (1536 tokens) are truncated with a warning.

From the command line, where `MODEL` is a saved directory, a training run directory or a Hub repo
id:

```bash
python -m gistgraph ask MODEL "Who founded the company?" --context-file report.txt --ratio 4 --verbose
python -m gistgraph export runs/m4_graph models/m4_graph   # training run -> shareable model
```

`GistGraph.from_run(run_dir)` loads a training run directly. `save_pretrained(dir)` writes
`config.json`, `compressor.safetensors` and a model card, the format `from_pretrained` reads, and
`push_to_hub(repo_id)` uploads it.

## Run experiments locally

Experiments are YAML files in `configs/experiments/`; any value can be overridden on the command
line (`a.b=value`). Each run writes its resolved config, logs, weights and `results.jsonl` to its
`out_dir`, and training resumes from its last checkpoint. Configs are validated when loaded, so
typos and out-of-range values fail with a clear error.

```bash
python -m gistgraph reproduce m1                      # non-learned baselines, all datasets
python -m gistgraph reproduce m2                      # flat-slot compressor
python -m gistgraph reproduce m3                      # routed, fixed and adaptive
python -m gistgraph reproduce m4                      # full graph model
python -m gistgraph reproduce m5                      # ablations and streaming
python -m gistgraph train --config configs/experiments/m4_graph.yaml train.steps=2000
python -m gistgraph probe --config configs/experiments/m4_graph.yaml     # edge-alignment probe
python -m gistgraph report runs/*                     # docs/results/results.md and figures
```

- Checkpoints, teacher caches and result rows are tied to the config that produced them. Resuming
  into an `out_dir` trained with a different config is refused; pass `--force` to `train` to
  resume anyway. Changing only the number of steps or evaluation settings is allowed.
- Steps with a non-finite loss or gradient are skipped, and training stops after 20 in a row.
  Checkpoints are written atomically, so an interrupted save never leaves a broken file.
- `--device` defaults to `cuda` when available, else `cpu`.
- `reproduce` reads `configs/`, which is not packaged, so run it from a checkout.
- `scripts/run_experiments.py` runs several configs in sequence, skips finished ones and exits
  non-zero if any failed; see [docs/design/m5.md](docs/design/m5.md) for the full command list.

## Milestones and design notes

| | Scope | Design notes |
|---|---|---|
| M0 | Scaffolding, config system, CI | [m0](docs/design/m0.md) |
| M1 | Evaluation harness and non-learned baselines | [m1](docs/design/m1.md) |
| M2 | Flat-slot compressor with distillation | [m2](docs/design/m2.md) |
| M3 | Routed writing and rate penalty | [m3](docs/design/m3.md) |
| M4 | Edge induction and message passing | [m4](docs/design/m4.md) |
| M5 | Streaming, ablations, analysis | [m5](docs/design/m5.md) |

Each note explains the design, the maths and why each choice was made.

## Layout

```
src/gistgraph/
  __main__.py          `python -m gistgraph` entry point (see cli.py)
  api.py               GistGraph inference API (load, compress, answer, save, publish)
  cli.py               command-line interface
  config.py            YAML config schema, inheritance, overrides and validation
  data/                datasets, prompts, teacher cache
  llm/frozen.py        frozen LM that can read soft prompts
  baselines/           truncation, extractive, LLMLingua-2
  model/               encoder, writers, gates, routing, edges, message passing, streaming
  train/               losses, rate controller, trainer
  eval/                metrics, harness, analysis, plots, probes
  utils/               config fingerprints, atomic writes, seeding
configs/               base config and experiments
notebooks/             Colab notebook: train, evaluate, report, publish
scripts/               batch experiment runner
docs/                  design notes, related work, results
tests/                 unit and integration tests
```

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first; it covers the
setup, the test and commit conventions, and how results should be reported (negative results
included). Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems should be
reported privately, as described in [SECURITY.md](SECURITY.md). Notable changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## Citation

If you use this code, please cite it (GitHub's "Cite this repository" button reads
[CITATION.cff](CITATION.cff)):

```bibtex
@software{deka_gistgraph,
  author = {Deka, Dhrrishit V},
  title  = {Gist Graph},
  year   = {2026},
  url    = {https://github.com/dhrrishitvdeka/gistgraph}
}
```

## Author

Dhrrishit V Deka ([@dhrrishitvdeka](https://github.com/dhrrishitvdeka)).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
