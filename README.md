# Gist Graph

Compress a long text context into a small **latent graph** that a frozen LLM reads as a soft prompt,
aiming to keep about 90% of downstream task performance at 3-5x compression.

> **Status: early research code (M0, scaffolding).** Nothing here has been evaluated yet. Results
> will be added only once they exist, and negative results will be reported as they are.

## Idea

1. A segment encoder turns the context into segment representations.
2. Segments write into K latent nodes through sparse top-k routing, so information-dense text can
   claim more capacity.
3. Sparse typed edges between nodes are induced by bilinear scoring, with no external parser.
4. A few message-passing layers encode multi-hop relations in the node states.
5. The nodes are projected into a frozen LLM's embedding space and read as a soft prompt.
6. New segments can be merged into the existing nodes (streaming memory).

Training distills the frozen LLM's output distribution on the full context into its output on the
compressed context, with a rate penalty on active nodes and edges.

## Hypotheses

- **H1** Graph structure beats flat latent slots at equal node count.
- **H2** Adaptive routing beats a fixed compression ratio.
- **H3** Gains concentrate on multi-hop tasks (HotpotQA, 2WikiMultihopQA) over single-hop (SQuAD).
- **H4** Incremental compression retains quality relative to one-shot compression.

See [docs/related_work.md](docs/related_work.md) for prior work and how this project differs.

## Install (development)

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

Experiments are described by YAML files in `configs/`. Any value can be overridden on the command
line with `a.b=value`.

## Roadmap

| Milestone | Scope |
|---|---|
| M0 | Scaffolding, config system, CI |
| M1 | Evaluation harness and non-learned baselines |
| M2 | Flat-slot compressor with distillation |
| M3 | Routed writing and rate penalty |
| M4 | Edge induction and message passing |
| M5 | Streaming, full ablations, write-up |

Design notes for each milestone live in `docs/design/`.

## License

Apache-2.0. See [LICENSE](LICENSE).
