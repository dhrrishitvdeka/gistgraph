"""Compressor assembly. ``build_compressor`` picks the variant named in the config."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from gistgraph.config import Config
from gistgraph.model.edges import EdgeInducer, random_adjacency, rw_positional_encoding
from gistgraph.model.encoder import SegmentEncoder
from gistgraph.model.gates import HardConcreteGate
from gistgraph.model.gnn import RelationalMP
from gistgraph.model.projector import Projector
from gistgraph.model.routing import RoutedWriter
from gistgraph.model.streaming import MergeGate, stream_update
from gistgraph.model.writer import FlatSlotWriter, num_slots


@dataclass
class Memory:
    """What the LLM reads: ``embeds[b, :counts[b]]`` are the soft-prompt vectors of document b."""

    embeds: Tensor  # [B, K, d_llm]
    mask: Tensor  # [B, K] bool
    aux: dict = field(default_factory=dict)

    @property
    def counts(self) -> Tensor:
        return self.mask.sum(1)

    def prefix(self, i: int) -> Tensor:
        return self.embeds[i, self.mask[i]]


class FlatCompressor(nn.Module):
    """Segment encoder -> flat Perceiver-style slots -> projector. The M2 baseline."""

    def __init__(self, d_llm: int, embed_norm: float, k_max: int, cfg: Config):
        super().__init__()
        c = cfg.compressor
        self.k_max = k_max
        self.encoder = SegmentEncoder(
            d_llm, c.encoder.d, c.encoder.layers, seg_len=c.encoder.seg_len
        )
        self.writer = FlatSlotWriter(c.encoder.d, k_max, layers=c.encoder.layers)
        self.projector = Projector(c.encoder.d, d_llm, embed_norm)

    def forward(self, h: Tensor, mask: Tensor, ratio: float | Tensor, chunks: int = 1) -> Memory:
        if chunks > 1:
            raise ValueError("the flat compressor has no incremental update")
        b = h.shape[0]
        ratio_t = torch.as_tensor(ratio, dtype=torch.float32, device=h.device).expand(b)
        counts = num_slots(mask.sum(1), ratio_t, self.k_max)
        u, u_valid = self.encoder(h, mask)
        z, z_valid, aux = self.writer(u, u_valid, counts, ratio_t)
        return Memory(self.projector(z), z_valid, aux)


class RoutedCompressor(nn.Module):
    """Segment encoder -> routed sparse writing -> (optional) adaptive gates -> projector.

    With ``writer.adaptive`` false every document gets exactly ``N / ratio`` nodes (fixed rate).
    With it true the candidate pool is ``capacity_factor`` times larger and hard-concrete gates
    decide how many nodes stay open, so information-dense documents can keep more of them. The
    rate penalty (see the trainer) holds the *average* open fraction at ``1 / ratio``.
    """

    def __init__(self, d_llm: int, embed_norm: float, k_max: int, cfg: Config):
        super().__init__()
        c = cfg.compressor
        self.k_max, self.adaptive, self.cf = k_max, c.writer.adaptive, c.writer.capacity_factor
        d = c.encoder.d
        self.encoder = SegmentEncoder(d_llm, d, c.encoder.layers, seg_len=c.encoder.seg_len)
        self.writer = RoutedWriter(
            d, k_max, c.writer.top_k, c.encoder.layers, c.writer.router_noise
        )
        self.gate = HardConcreteGate(d, c.writer.gate_init) if self.adaptive else None
        self.projector = Projector(d, d_llm, embed_norm)
        # learned merge correction exists only for models trained with streaming; without it
        # the merge is the plain mass-weighted running average
        self.merge = MergeGate(d) if c.streaming.enabled else None

    def _nodes_stream(self, h: Tensor, mask: Tensor, ratio: float | Tensor, chunks: int):
        """Build the node memory incrementally from ``chunks`` consecutive pieces of the context.

        Each piece is encoded on its own (it cannot see later text) and folded into the existing
        nodes by :func:`stream_update`. The node budget grows with the text seen so far, so the
        final memory has the same size as in the one-shot case.
        """
        b, n, _ = h.shape
        ratio_t = torch.as_tensor(ratio, dtype=torch.float32, device=h.device).expand(b)
        n_tokens = mask.sum(1)
        pool_ratio = ratio_t / self.cf if self.adaptive else ratio_t
        seg = self.encoder.seg_len
        step = math.ceil(math.ceil(n / chunks) / seg) * seg  # chunk length, a multiple of seg_len
        total_units = torch.ceil(n_tokens.float() / seg).clamp(min=1)
        state, lbs, z_valid = None, [], None
        for lo in range(0, n, step):
            hc, mc = h[:, lo : lo + step], mask[:, lo : lo + step]
            has = mc.any(1)
            safe = mc.clone()
            safe[:, 0] |= ~has  # keep the encoder away from fully masked rows
            u, u_valid = self.encoder(hc, safe, pos_offset=lo // seg)
            u_valid = u_valid & has[:, None]
            seen = torch.clamp(n_tokens, max=lo + step)
            counts = num_slots(seen, pool_ratio, self.k_max)
            pos = (lo // seg + torch.arange(u.shape[1], device=h.device))[None, :] / total_units[
                :, None
            ]
            state, lb, z_valid = stream_update(
                self.writer, self.merge, state, u, u_valid, pos, counts, ratio_t
            )
            lbs.append(lb)
        keys = self.writer.node_keys(state.z.shape[1], ratio_t)
        z = self.writer.finalize(state.z, keys, z_valid)
        seen_any = state.mass > 1e-6
        centroid = torch.where(
            seen_any, state.pos_sum / state.mass.clamp(min=1e-6), torch.full_like(state.mass, 2.0)
        )
        aux = {
            "lb": torch.stack(lbs).mean(),
            "centroid": centroid,
            "mass": state.mass,
            "n_tokens": n_tokens.float(),
        }
        return z, z_valid, aux, ratio_t

    def _nodes(self, h: Tensor, mask: Tensor, ratio: float | Tensor, chunks: int = 1):
        """Encode and write. Returns node states, node validity, aux and the ratio tensor."""
        if chunks > 1:
            return self._nodes_stream(h, mask, ratio, chunks)
        b = h.shape[0]
        ratio_t = torch.as_tensor(ratio, dtype=torch.float32, device=h.device).expand(b)
        n_tokens = mask.sum(1)
        pool_ratio = ratio_t / self.cf if self.adaptive else ratio_t
        counts = num_slots(n_tokens, pool_ratio, self.k_max)
        u, u_valid = self.encoder(h, mask)
        z, z_valid, aux = self.writer(u, u_valid, counts, ratio_t)
        aux["n_tokens"] = n_tokens.float()
        return z, z_valid, aux, ratio_t

    def _open(self, z: Tensor, z_valid: Tensor, aux: dict) -> Tensor:
        """Gate values ``[B, K]``; fixed-rate models keep every valid node fully open."""
        if not self.adaptive:
            return z_valid.float()
        gate, expected_open = self.gate(z, z_valid)
        aux["exp_active"] = expected_open
        # never let a document end up with an empty memory: reopen its best node
        empty = gate.sum(1) == 0
        if empty.any():
            best = self.gate.log_alpha(z).masked_fill(~z_valid, float("-inf")).argmax(1)
            rescue = torch.zeros_like(gate).scatter(1, best[:, None], 1.0)
            gate = torch.where(empty[:, None], rescue.detach(), gate)
        return gate

    def forward(self, h: Tensor, mask: Tensor, ratio: float | Tensor, chunks: int = 1) -> Memory:
        z, z_valid, aux, _ = self._nodes(h, mask, ratio, chunks)
        gate = self._open(z, z_valid, aux)
        return self._finish(z, gate, aux)

    def _finish(self, z: Tensor, gate: Tensor, aux: dict, pe: Tensor | None = None):
        embeds = self.projector(z, pe) * gate.unsqueeze(-1)
        active = gate > 0
        key = aux["centroid"]
        order = torch.where(active, key, torch.full_like(key, 3.0)).argsort(dim=1, stable=True)
        embeds = embeds.gather(1, order[..., None].expand(-1, -1, embeds.shape[-1]))
        active = active.gather(1, order)
        aux["active"] = active.sum(1)
        aux["gate"] = gate.detach()  # in the original node order, like aux["adj"] and aux["write"]
        return Memory(embeds, active, aux)


RANDOM_GRAPH_SEED = 1234


def _control_generator(valid: Tensor) -> torch.Generator:
    """Generator for the random-graph control, seeded by the input size so that the same shape
    always gets the same graph (across calls, evals and processes)."""
    gen = torch.Generator(device=valid.device)
    gen.manual_seed(RANDOM_GRAPH_SEED + valid.shape[0] * 100_003 + valid.shape[1])
    return gen


class GraphCompressor(RoutedCompressor):
    """Routed nodes + induced typed edges + message passing + graph positional bias.

    ``edges.enabled`` and ``gnn.enabled`` switch the two structural pieces independently, which
    gives the controls used to test whether structure helps (H1):

    * edges on, GNN on: the full model;
    * edges off, GNN on: same parameters, but messages are empty (parameter-matched control);
    * ``edges.mode: random``: random sparse graph of equal degree, so content-blind structure;
    * both off: reduces to the routed model.
    """

    def __init__(self, d_llm: int, embed_norm: float, k_max: int, cfg: Config):
        super().__init__(d_llm, embed_norm, k_max, cfg)
        c = cfg.compressor
        d = c.encoder.d
        self.ec, self.pe_steps = c.edges, (c.projector.pe_steps if c.projector.pe == "rw" else 0)
        if c.edges.mode not in ("learned", "random"):
            raise ValueError(f"unknown edges.mode {c.edges.mode!r} (expected learned | random)")
        self.edges = None
        if c.edges.enabled and c.edges.mode == "learned":
            self.edges = EdgeInducer(
                d, c.edges.relations, c.edges.rank, c.edges.sparsifier, c.edges.max_deg
            )
        self.gnn = RelationalMP(d, c.edges.relations, c.gnn.layers) if c.gnn.enabled else None
        self.projector = Projector(d, d_llm, embed_norm, pe_dim=self.pe_steps)

    def forward(self, h: Tensor, mask: Tensor, ratio: float | Tensor, chunks: int = 1) -> Memory:
        z, z_valid, aux, _ = self._nodes(h, mask, ratio, chunks)
        gate = self._open(z, z_valid, aux)
        active = gate > 0
        z = z * gate.unsqueeze(-1)
        adj = None
        if self.edges is not None:
            adj, edge_aux = self.edges(z, active)
            aux.update(edge_aux)
        elif self.ec.enabled and self.ec.mode == "random":  # random-graph control
            adj = random_adjacency(
                active, self.ec.relations, self.ec.max_deg, _control_generator(active)
            )
        if self.gnn is not None:
            z = self.gnn(z, adj, active)
        pe = None
        if self.pe_steps:
            if adj is None:
                pe = torch.zeros(*z.shape[:2], self.pe_steps, device=z.device)
            else:
                pe = rw_positional_encoding(adj, self.pe_steps)
        if adj is not None:
            aux["adj"] = adj.detach()
        return self._finish(z, gate, aux, pe)


def max_slots(cfg: Config) -> int:
    """Largest slot count any training or eval ratio can ask for."""
    lowest = min([*cfg.train.ratios, cfg.compressor.ratio])
    pool = cfg.compressor.writer.capacity_factor if cfg.compressor.writer.adaptive else 1.0
    return math.ceil(pool * cfg.data.max_ctx_tokens / lowest)


def build_compressor(cfg: Config, d_llm: int, embed_norm: float) -> nn.Module:
    kind = cfg.compressor.type
    if kind == "flat":
        return FlatCompressor(d_llm, embed_norm, max_slots(cfg), cfg)
    if kind == "routed":
        return RoutedCompressor(d_llm, embed_norm, max_slots(cfg), cfg)
    if kind == "graph":
        return GraphCompressor(d_llm, embed_norm, max_slots(cfg), cfg)
    raise KeyError(f"unknown compressor type {kind!r}")
