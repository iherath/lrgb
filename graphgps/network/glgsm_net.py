"""GraphGym network wrapping GenLGSM (GLGSM) for the LRGB harness.

Pipeline:  Atom node encoder  ->  GenLGSM body (node-level output)  ->  decoder head.

Matches the LGSM peptides protocol (paper Table 9 / E.3): node features + graph
structure only -- no positional encoding, no edge/bond features. GLGSM's forward is
(x, edge_index, batch), so this is exact by construction.

GenLGSM lives in the sibling Graph-SSM repo. Put it on PYTHONPATH (see README_glgsm.md);
the sys.path fallback covers the side-by-side checkout layout (.../lrgb, .../Graph-SSM).
"""

import os
import sys

import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder
from torch_geometric.graphgym.register import register_network

try:
    from glgsm_model import GenLGSMModel
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../Graph-SSM"))
    from glgsm_model import GenLGSMModel


class GLGSMNet(torch.nn.Module):
    """Atom encoder -> GenLGSM (per-node embeddings) -> GraphGym pooling+decoder head."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)      # Atom encoder -> cfg.gnn.dim_inner
        dim_in = self.encoder.dim_in
        assert cfg.gnn.dim_inner == dim_in, \
            "cfg.gnn.dim_inner must equal the Atom encoder output dim."

        # Node-level body: returns (N_total, dim_inner); the head pools + decodes to dim_out.
        self.gnn = GenLGSMModel(
            in_dim=dim_in,
            d_model=cfg.gnn.dim_inner,
            d_state=cfg.glgsm.d_state,
            out_dim=cfg.gnn.dim_inner,
            max_hops=cfg.glgsm.max_hops,          # LGSM "seq. length"
            mode=cfg.glgsm.mode,                  # LGSM "seq. type"
            num_blocks=cfg.glgsm.num_blocks,      # LGSM "num. blocks"
            task_type="node",
            dropout=cfg.gnn.dropout,
            window_size=cfg.glgsm.window_size,
            hyper_hidden_dim=cfg.glgsm.hyper_hidden_dim,
            batched=cfg.glgsm.batched,
        )

        GNNHead = register.head_dict[cfg.gnn.head]  # graph-pooling + post-MP decoder
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)                                    # batch.x -> (N, dim_inner)
        batch.x = self.gnn(batch.x, batch.edge_index, batch.batch)     # (N, dim_inner)
        batch = self.post_mp(batch)                                    # pool + decode -> (B, dim_out)
        return batch


register_network('glgsm', GLGSMNet)
