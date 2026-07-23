"""GraphGym network wrapping GenLGSM (GLGSM) for the LRGB harness.

Pipeline:  Atom node encoder  ->  GenLGSM with its own graph decoder.

Uses GenLGSMModel's task_type='graph' head -- sum+max+mean pooling concatenated, then
Linear -> GELU -> Linear -- which is exactly the LGSM paper's decoder (Table 9 / E.3),
instead of GraphGym's single-pool head. Otherwise matches the LGSM protocol: node features +
graph structure only, no positional encoding, no edge/bond features (GLGSM's forward is
(x, edge_index, batch), so this is exact by construction).

GenLGSM lives in the sibling Graph-SSM repo. Put it on PYTHONPATH (see README_glgsm.md);
the sys.path fallback covers the side-by-side checkout layout (.../lrgb, .../Graph-SSM).
"""

import os
import sys

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder
from torch_geometric.graphgym.register import register_network

try:
    from glgsm_model import GenLGSMModel
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../Graph-SSM"))
    from glgsm_model import GenLGSMModel


class GLGSMNet(torch.nn.Module):
    """Atom encoder -> GenLGSM graph model (3-pool concat + MLP decoder, LGSM-style)."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)      # Atom encoder -> cfg.gnn.dim_inner
        dim_in = self.encoder.dim_in
        assert cfg.gnn.dim_inner == dim_in, \
            "cfg.gnn.dim_inner must equal the Atom encoder output dim."

        # task_type='graph': GenLGSM pools (sum+max+mean concat) and runs its own decoder MLP
        # to dim_out -- the LGSM paper's decoder head, so no separate GraphGym head is used.
        self.gnn = GenLGSMModel(
            in_dim=dim_in,
            d_model=cfg.gnn.dim_inner,
            d_state=cfg.glgsm.d_state,
            out_dim=dim_out,
            max_hops=cfg.glgsm.max_hops,          # LGSM "seq. length"
            mode=cfg.glgsm.mode,                  # LGSM "seq. type"
            num_blocks=cfg.glgsm.num_blocks,      # LGSM "num. blocks"
            task_type="graph",
            dropout=cfg.gnn.dropout,
            window_size=cfg.glgsm.window_size,
            hyper_hidden_dim=cfg.glgsm.hyper_hidden_dim,
            batched=cfg.glgsm.batched,
        )

    def forward(self, batch):
        batch = self.encoder(batch)                              # batch.x -> (N, dim_inner)
        pred = self.gnn(batch.x, batch.edge_index, batch.batch)  # (B, dim_out) via LGSM decoder
        return pred, batch.y


register_network('glgsm', GLGSMNet)
