"""Config group for the GLGSM (GenLGSM) network.

The three LGSM sweep knobs map here:
    mode      = seq. type    (option1/option2/path_b/lgsm_adj/lgsm_nbt/hyper)
    max_hops  = seq. length
    num_blocks= num. blocks
Defaults reflect LGSM's peptides finding (E.3): shorter sequences + 'adj' regularize best.
"""

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_glgsm(cfg):
    cfg.glgsm = CN()
    cfg.glgsm.mode = "lgsm_adj"       # seq. type
    cfg.glgsm.max_hops = 8            # seq. length L
    cfg.glgsm.num_blocks = 2          # processing depth (num. blocks)
    cfg.glgsm.d_state = 16            # Mamba2 SSM state dim
    cfg.glgsm.window_size = 2         # memory depth (only used by mode='hyper')
    cfg.glgsm.hyper_hidden_dim = 64   # hypernetwork hidden dim (only used by mode='hyper')
    cfg.glgsm.batched = False         # padded-batch vectorized layers


register_config('glgsm', set_cfg_glgsm)
