# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

### Added
- Config system: typed YAML schema with inheritance, dotted overrides and strict validation.
- Evaluation harness with SQuAD-style EM and F1, achieved-ratio reporting and retention tables.
- Non-learned baselines: truncation, extractive (lead, TextRank, BM25) and LLMLingua-2.
- Flat-slot Perceiver-style compressor trained by distillation from the full-context LLM, with a
  reconstruction warm-up, a teacher cache and resumable training.
- Routed sparse writing, hard-concrete node gates and an augmented-Lagrangian rate controller.
- Entmax edge induction, relational message passing and a random-walk positional bias, with
  no-edge and random-graph controls.
- Streaming memory with a mass-weighted merge gate.
- Paired-bootstrap analysis for four hypotheses, an edge-alignment probe, figures and a results
  builder.
- Design notes for every milestone in `docs/design/`.

### Status
- No compressor has been fully trained yet, so no performance claim is made for the method.
