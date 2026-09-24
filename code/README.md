# Code and Fixed Experimental Configurations

This is the actual implementation used in the archived experiments, not a
simplified reimplementation. `sources/primary/` is the frozen R2/R3 training
code; `sources/primary_r1/` preserves R1. Keeping both is necessary to reproduce
the recorded executions. `configs/runs.json` resolves each run to its source,
environment, exact command arguments, data image directory and result files.

## Source Layout

| Path under `sources/` | Role |
| --- | --- |
| `primary/factorgrad/factorgrad/` | Structural-gradient controller and objective |
| `primary/fastgs-factorgrad/` | Main training, loaders, preprocessing, rendering and metrics |
| `primary_r1/` | R1 and sensitivity snapshot |
| `baselines/` | Locally reproduced comparison implementations, excluding the separately supplied Mini-Splatting source |
| `transfer/`, `shorter_transfer/` | Additional-host integrations |
| `mechanism/` | Instrumented implementation used for gradient/timing evidence |

The third-party SkipGS plugin is not bundled. For fresh FastGS-based training,
set `FACTORGRAD_SKIPGS_SOURCE` to a separately obtained checkout containing
`skipgs/__init__.py`. To run the Mini-Splatting baseline, set
`FACTORGRAD_MINI_SOURCE` to an authorized checkout containing `ms/`.
Retain each external project's own license and attribution.

The host folders include their CUDA/C++ extension sources. Full Gaussian PLY
stores position, spherical harmonics, opacity, scale and rotation. This is an
inference model; it is not an Adam-state resume checkpoint. Models are delivered
separately from the submission archive.

## Dependencies

Linux, Conda, NVIDIA GPU and a matching CUDA build toolkit are required for
training/evaluation. The primary recorded environment is Python 3.7.13,
PyTorch 1.12.1, torchvision 0.13.1 and CUDA runtime 11.6. The recorded GPU is
RTX 3090. Conda explicit locks, exact pip versions, extension inventories and
version reports for all eight environments are in `environments/`.

LeGS uses its separate Python 3.9 / PyTorch 2.8.0+cu128 environment; Shorter-GS
uses Python 3.10 / PyTorch 2.0.1+cu117. Do not merge the hosts into one environment:
their rasterizers use overlapping Python import names.

The recorded CUDA runtime and matching build toolkit differ by host:

| Environments | CUDA |
| --- | --- |
| `fastgs`, `factorgrad-vanilla`, `factorgrad-mini`, `factorgrad-speedy`, `factorgrad-taming`, `factorgrad-dash` | 11.6 |
| `LeGS` | 12.8 |
| `factorgrad-shorter` | 11.7 |

For a new primary environment, run the following from the `code/` directory
(the directory containing this README), with Conda on PATH and a matching
CUDA toolkit with `nvcc` installed:

```bash
export CUDA_HOME=/path/to/cuda-11.6
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/setup_environment.sh fastgs
```

The script refuses to overwrite an existing environment. Other valid names
are `factorgrad-vanilla`, `factorgrad-mini`, `factorgrad-speedy`,
`factorgrad-taming`, `factorgrad-dash`, `LeGS`, and `factorgrad-shorter`.
For `LeGS` or `factorgrad-shorter`, set `CUDA_HOME` and `PATH` to the matching
12.8 or 11.7 toolkit before building that environment.
Dependency installation and first-time pretrained metric-network loading
require network access or populated local package/model caches. Neither full
datasets nor system CUDA toolkits are embedded in the submission ZIP.

## Data and Commands

```text
datasets/
  mipnerf360/{bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai}/
  tnt/{train,truck}/
  db/{drjohnson,playroom}/
```

Every scene uses the original COLMAP `sparse/0` files and the recorded `images`,
`images_2` or `images_4` directory. The implementation retains the original test
split and camera order. Host `convert.py` files support preprocessing new data;
rerunning COLMAP is not a substitute for the fixed paper inputs.

Run the three scripts in the parent directory: `reproduce_tables.sh` for saved
metrics, `train_and_evaluate.sh` for fixed-configuration training and evaluation,
or `evaluate_models.sh` for rerendering supplied archived weights. Use `--suite
ablation`, `--suite sensitivity`, or `--suite transfer` for the other protocols.
`--action list --suite all` lists run IDs. Output directories are separate from
the archived evidence; `--fps` explicitly adds a new render-only FPS measurement.

Training starts from dataset initialization. Repeated training can differ
numerically, and timing depends on hardware and load. Saved metric aggregation
is exact; bitwise reproduction of a new training trajectory is not promised.

## Checks

```bash
python3 -m unittest discover -s tests -v
```

All included runtime source files can be checked against
`configs/source_equivalence.json`. The supplemental scripts only relocate input
and output paths, replay saved options, and validate evaluation output. No
algorithm change was made during packaging.
