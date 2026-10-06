# Related work

This page records the prior-work check done before building Gist Graph (GLM-style graph-latent
context compression). Entries marked **(checked)** were looked up in a search or abstract at the
time of writing. Entries marked **(from memory)** are well-known works summarised without
re-reading; verify details before citing them in the paper.

## 1. Soft-prompt / latent context compression (flat memory)

| Work | What it does | Relation to Gist Graph |
|---|---|---|
| Gist tokens (Mu et al., 2023) **(from memory)** | The LM learns to compress a prompt into a few gist tokens through attention masking | Flat slots, and the LM itself is fine-tuned |
| AutoCompressors (Chevalier et al., 2023) **(from memory)** | Recursively produced summary vectors over segments | Streaming-like, but flat and with a fine-tuned LM |
| ICAE (Ge et al., 2024) **(from memory)** | LoRA encoder produces memory slots for a frozen decoder, trained with autoencoding then fine-tuning | Closest flat-slot design. Used as the M2 baseline style |
| xRAG (2024) **(checked)** | Extreme compression of a retrieved document into one token | Lags behind on multi-hop tasks such as HotpotQA |
| PCC, "Pretraining Context Compressor ... Embedding-Based Memory" (ACL 2025) **(checked)** | Pretrained embedding-based compressor | Flat memory |
| In-Context Former (2024) **(checked)** | Lightweight cross-attention compressor | Flat memory |
| Latent Context Compilation (arXiv 2602.21221) **(checked)** | A disposable LoRA "compiles" context into buffer tokens, up to 16x on an 8B model | Flat tokens, a different training approach |
| LatentPress (arXiv 2609.01507) **(checked)** | A small writer plus a 4-26M-parameter adapter writes memory tokens for a frozen decoder, 4-16x | Flat memory, and the closest recent frozen-decoder recipe |
| CMC (arXiv 2609.25537) **(checked)** | Chunked compression into memory embeddings projected into a frozen decoder, reports HotpotQA | Its "graph denoising" is a cosine-similarity filter on token embeddings, not a learned latent graph |
| Latent Personal Memory (arXiv 2606.20911) **(checked)** | A matrix of latent slots mapped by cross-attention to a soft prompt | Flat slots |

## 2. Adaptive compression rate

| Work | What it does | Relation |
|---|---|---|
| ATACompressor (arXiv 2602.03226) **(checked)** | A controller adapts the rate to the length of task-relevant content, using a selective encoder | Adaptive but query-aware and flat. Ours is query-agnostic, with routed capacity and an L0 rate penalty |
| Meta-Soft (arXiv 2605.22337) **(checked)** | Composable meta-tokens for KV-cache compression | Different mechanism |

## 3. Recurrent and slot memory

RMT (Bulatov et al., 2022) and the Compressive Transformer (Rae et al., 2020) **(from memory)** keep
recurrent or compressed memory. G-MemLLM (arXiv 2602.00015) and "Trained Persistent Memory for
Frozen Decoder-Only LLMs" (arXiv 2603.22329, includes sparse top-k slot writes) **(checked)** are
slot-memory designs. None of them models relations between slots. These are the reference points
for the streaming experiment (H4).

## 4. Cross-attention bottlenecks

Perceiver / Perceiver IO (Jaegle et al.) and the BLIP-2 Q-Former (Li et al., 2023) **(from memory)**
use learned latent queries that cross-attend to an input. This is the template for the flat writer
in M2. Latent Bridges (arXiv 2606.28916) **(checked)** applies a Perceiver-style resampler to
soft tokens for table QA.

## 5. Hard-prompt compression (text in, text out)

| Work | Notes |
|---|---|
| LLMLingua / LongLLMLingua **(from memory)** | Token pruning scored by a small LM's information entropy |
| LLMLingua-2 (arXiv 2403.12968) **(checked)** | Token classification with a bidirectional encoder, trained on data distilled from a larger LLM. Reported 2x-5x compression. M1 baseline |
| RECOMP (Xu et al., 2023) **(from memory)** | Extractive and abstractive compressors for retrieved documents |
| EDU-based compression (arXiv 2512.14244) **(checked)** | Compresses over elementary discourse units |

## 6. Graphs and LLMs

| Work | Notes |
|---|---|
| GraphRAG (Edge et al., 2024) **(from memory)** | Builds an explicit entity/knowledge graph with an LLM and summarises communities. Needs extraction |
| GraphToken (arXiv 2402.05862) **(checked)** | A GNN encodes a *given* graph into soft tokens for a frozen LLM. Closest in mechanism, but the graph is an input |
| GraphPrompter, GraphTokenLM **(checked, via search snippets)** | Same family |
| ReCAP context graphs (arXiv 2609.40118) **(checked)** | Stores attention-derived importance scores and dependency links in a persistent graph to select messages |

## Novelty assessment

No work found in this search combines all of the following: (a) routed sparse writing of text
into latent nodes with adaptive capacity, (b) typed edges induced from node states with no parser
or entity extraction, (c) message passing over that latent graph, (d) a soft prompt for a frozen
decoder trained by KL distillation, and (e) incremental node updates. This is a result of a
limited search, not proof of absence.

The nearest neighbours are GraphToken (graph given, not induced) and flat-slot compressors
(LatentPress, ICAE, CMC). The contribution is therefore framed as an empirical test of whether
**induced structure helps at equal node count and equal parameter count (H1)**, and where it
helps (H3), with matched flat, edge-free and random-graph controls. A null result on H1 would
still be a useful and publishable finding.
