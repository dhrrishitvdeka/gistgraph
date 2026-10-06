# Documentation

| Page | What it covers |
|---|---|
| [related_work.md](related_work.md) | Prior work (soft-prompt compressors, adaptive rates, graph and LLM methods), what is and is not novel here |
| [design/m0.md](design/m0.md) | Scaffolding: config system, seeding, CI, repository hygiene |
| [design/m1.md](design/m1.md) | Evaluation harness, metrics, non-learned baselines, partial baseline results |
| [design/m2.md](design/m2.md) | Flat-slot compressor, distillation objective, training loop |
| [design/m3.md](design/m3.md) | Routed writing, hard-concrete gates, rate controller |
| [design/m4.md](design/m4.md) | Edge induction, message passing, structural controls |
| [design/m5.md](design/m5.md) | Streaming memory, ablations, analysis, structure probe, experiment status |
| [results/results.md](results/results.md) | Generated results tables and figures (see its header for what it covers) |

Each design note explains what was built, the maths, and *why* each choice was made, so the code can
be understood and defended without reading it all. They describe the code as it is: if you change
behaviour, change the note in the same commit.

New to the project? Read the top-level [README](../README.md) for the idea and the hypotheses, then
`design/m2.md` through `design/m5.md` in order.
