# Running GLGSM on LRGB Peptides (comparable to LGSM) — RunPod setup

Goal: evaluate **GLGSM (GenLGSM)** on **peptides-func** and **peptides-struct** using the same
protocol LGSM followed (Tönshoff et al., 2024): **AdamW + cosine schedule, 250 epochs, batch 200,
decoder head, ~500k param budget**, with a small sweep over **(seq. type, seq. length, num. blocks)**.

Per the LGSM paper (§E.3, Table 9), the protocol uses **no positional encoding** and **no edge/bond
features** — GLGSM matches this by construction (its forward is `(x, edge_index, batch)` only).

**Targets to beat (LGSM, 3 seeds):** Peptides-func AP `66.85±1.36`, Peptides-struct MAE `0.2470±0.0019`.

---

## Version note (read first)

The lrgb repo's own README pins **torch 1.9 / pyg 2.0.2**. We **cannot** use that: GLGSM's Mamba2
blocks need **mamba-ssm**, which requires **torch 2.x**. So this env uses the modern stack
(**torch 2.4.1+cu124 / pyg ≥2.4 / mamba-ssm 2.2.2**). GraphGym ships inside `torch_geometric` on
pyg ≥2.x. A few GraphGym call sites in this repo were written for pyg 2.0.2 and may need small
compatibility patches on pyg 2.4 — those live in the GLGSM integration (network + configs), not here.

## Prerequisites on RunPod

- A CUDA 12.x GPU pod (e.g. the "PyTorch 2.x / CUDA 12.4" template, or a bare CUDA 12.4 image).
- `conda`/`miniconda` available (install miniconda if the image lacks it).
- Both repos cloned side by side, e.g. under `/workspace`:
  ```
  /workspace/lrgb          # this repo (GraphGym harness)
  /workspace/Graph-SSM     # GLGSM model code (glgsm_model.py, glgsm_layer.py, ...)
  ```

## Setup

**1. Create and activate the env** (everything except mamba-ssm):
```
conda env create -f environment_glgsm.yml && conda activate glgsm-lrgb
```

**2. Verify the GPU torch install:**
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**3. Install mamba-ssm + causal-conv1d** — must be a separate step, `--no-build-isolation`, because
they compile against the torch installed in step 1:
```
pip install "causal-conv1d>=1.4.0,<1.5.0" "mamba-ssm==2.2.2" --no-build-isolation
```

**4. Put GLGSM on the import path** so the harness can `import glgsm_model`:
```
export PYTHONPATH=/workspace/Graph-SSM:$PYTHONPATH
```
(The GLGSM network file also does a `sys.path.insert` fallback, but exporting this is the reliable path.)

**5. Sanity-check imports:**
```
python -c "import torch_geometric.graphgym, mamba_ssm; from glgsm_model import GenLGSMModel; print('ok')"
```

## Running (after the GLGSM integration is in place)

```
python main.py --cfg configs/GLGSM/peptides-func-GLGSM.yaml   wandb.use False
python main.py --cfg configs/GLGSM/peptides-struct-GLGSM.yaml wandb.use False
```
Peptides data downloads automatically on first run. Drop `wandb.use False` to log to Weights & Biases.

## Hyperparameter sweep (LGSM §E.3 guidance)

LGSM found that **longer sequences did not help** on peptides — what mattered was **regularization**.
Bias the sweep toward short sequences + the `adj` sequence type, and expect to tune dropout/weight-decay:

- **seq. type** (`glgsm.mode`): `lgsm_adj` (favored), `lgsm_nbt`, `hyper`
- **seq. length** (`glgsm.max_hops`): short, e.g. `4`, `8`, `16`
- **num. blocks** (`glgsm.num_blocks`): `1`, `2`, `4`
- Keep total params near the **~500k** LRGB budget (tune `gnn.dim_inner`); GraphGym prints the count at startup.
