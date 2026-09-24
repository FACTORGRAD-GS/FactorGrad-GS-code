# FactorGrad-GS Supplementary Code

This is the code repository for the anonymous manuscript's
[project page](https://factorgrad-gs.github.io/). It contains the
implementation and saved numeric results as a compact GitHub edition of the
supplementary materials. The full submission archive also contains training
logs and a video; those large files are not included in this repository.
The third-party SkipGS plugin and Mini-Splatting source are supplied separately
by the reviewer when their runs are needed; see **External Source Dependencies**
below.

Its folders are:

| Folder | Contents |
| --- | --- |
| `code/` | Actual training, data loading, preprocessing, rendering, metrics, CUDA sources, fixed configurations, and environment locks |
| `results/results_ours.xlsx` | FactorGrad integrations, ablations, sensitivity and boundary results, one row per scene execution |
| `results/baselines/results_baselines.xlsx` | Reproduced baseline results, kept separately |
| `results/evidence/` | Original numeric result files and evaluation configurations, with local paths redacted |
| `results/reference_tables/` | Repeated-execution and mechanism analysis source tables |

The omitted `logs/` and `visualization/` folders remain in the full
supplementary archive. Selected visual comparisons appear on the separate
project page.

## CUDA and Environment Setup

The primary experiments used Linux, an NVIDIA RTX 3090, Python 3.7.13,
PyTorch 1.12.1 and CUDA 11.6. The exact Linux Conda package list is in
[`code/environments/fastgs/conda-explicit.txt`](code/environments/fastgs/conda-explicit.txt),
with pip versions in
[`requirements.txt`](code/environments/fastgs/requirements.txt) and the
recorded version report in
[`versions.json`](code/environments/fastgs/versions.json).
Install a CUDA 11.6 toolkit with `nvcc` separately, then run these commands
from the repository root with Conda on `PATH`:

```bash
export CUDA_HOME=/path/to/cuda-11.6
export PATH="$CUDA_HOME/bin:$PATH"
bash code/scripts/setup_environment.sh fastgs
```

The script creates the `fastgs` Conda environment and builds the included CUDA
extensions. Use the same Conda installation as `--conda-root` in the training
and evaluation commands below. Other hosts require separate environments;
LeGS uses CUDA 12.8 and `factorgrad-shorter` uses CUDA 11.7. See
[`code/README.md`](code/README.md) and
[`code/environments/`](code/environments/) for their locks and build steps.
The system CUDA toolkits, datasets and external SkipGS/Mini-Splatting source
are not bundled. A clean installation of all eight environments has not been
verified during packaging.

## Read or Recompute Results

From the unpacked directory, this command regenerates the reported metric
aggregations from the saved per-scene result files, without a GPU:

```bash
bash reproduce_tables.sh
```

This is **recorded-result aggregation**, not a new training or rendering run.
The Excel files contain individual executions and scene-level mean/std sheets.
Missing historical measurements are marked `NR`, not replaced with zero.

## External Source Dependencies

The anonymous repository omits two separately maintained codebases:

- **SkipGS plugin:** required for fresh training of the FastGS-based main,
  ablation and mechanism runs. Obtain an unmodified SkipGS source checkout
  separately, preserve its license, and point to the directory containing
  `skipgs/__init__.py`:

  ```bash
  export FACTORGRAD_SKIPGS_SOURCE=/path/to/SkipGS
  ```

- **Mini-Splatting:** required only to freshly train or evaluate the
  Mini-Splatting baseline. Its source is not redistributed here. If you have
  a separately authorized checkout, point to the directory containing `ms/`:

  ```bash
  export FACTORGRAD_MINI_SOURCE=/path/to/mini-splatting
  ```

The saved `results/` remain available for recorded-result aggregation without
either source checkout. `--suite main` needs the SkipGS plugin for fresh
training; `--suite table1` and `--suite all` additionally need Mini-Splatting
for all recorded methods. The scripts fail with an explicit message if an
external source needed for a selected run is absent.

Full datasets, trained Gaussian PLY models and pretrained metric networks are
not embedded in this size-limited package. To recompute image metrics, supply
the archived model artifacts and original preprocessed datasets:

```bash
bash evaluate_models.sh --artifacts /path/to/artifacts \
  --data-root /path/to/datasets --conda-root /path/to/miniconda3
```

This command genuinely renders held-out views and computes PSNR, SSIM and
LPIPS; it fails when models are absent instead of printing cached metrics.
Dataset and environment setup are described in `code/README.md`.

To train and then evaluate with the recorded parameters:

```bash
bash train_and_evaluate.sh --data-root /path/to/datasets \
  --conda-root /path/to/miniconda3
```

The default training suite is the three main methods and their three recorded
executions. For a single scene, append
`--run-id primary/R3/full/bonsai`; no algorithm tuning is required.

## Evidence Scope

There are 632 archived scene executions: primary comparisons/ablations (312),
additional sensitivity/host runs (78), additional baselines (91), transfer
pairs (104), Shorter-GS transfer (26), Vanilla Full boundary runs (13), and
mechanism runs (8). Repeated execution IDs are kept separate. R1/R2/R3 are
fixed-seed repetitions, not different seeds.

Quality is averaged over scenes and then executions. Main-table training
process-wall time is summed over scenes and then averaged over executions.
Native training timers, external process-wall times, render-only FPS and
training peak device memory remain distinct fields. The transfer protocols
must not be pooled into the main timing protocol. Re-evaluating a final PLY
does not remeasure historical training time or training peak memory.

The main implementation uses FastGS Base, densification interval 500, 30K
iterations, probability 2/3, 768 stabilization steps and early start at 3K.
All scene-specific settings are stored in `code/configs/runs.json`.

## Training Curves (Full Supplementary Archive)

The full supplementary archive has 620 event files from 619 scene runs. `curves.csv.xz`
contains the first and last scalar point and every recorded multiple of 100
steps, with original values and wall times. `curve_intervals.csv.xz` adds
count, min, max and mean over **all original scalar points** in each 1000-step
interval. These are compact exports, not full-resolution event-file copies.
`logs/curve_export_inventory.json` records source point counts and tags in
that archive. These log files are not included in the GitHub edition.

The complete original TensorBoard events remain in the local research
archive. Thirteen archived runs have no TensorBoard file; no curve was
fabricated for them. Textual training progress is preserved separately.

To view an exported curve from the full archive in TensorBoard in the primary
environment:

```bash
python code/scripts/restore_curve_viewer.py \
  logs/train/primary/R3/full/bonsai/curves.csv.xz --output /tmp/fg_curves
tensorboard --logdir /tmp/fg_curves
```

## Visualization (Full Supplementary Archive)

The full archive's video uses real R2 Full renders and corresponding captured images from
all three datasets. It contains both training and held-out camera views and
is qualitative material, **not** a held-out metric calculation or a synthetic
360-degree trajectory. Its `visualization/scene_index.json` gives scene
boundaries and exact run IDs. The video is not included in this GitHub edition.
No generated image, interpolated view or retouching is used.

## Code Provenance and Validation

Included host, FactorGrad and CUDA implementation files are byte-identical to
the audited experimental source snapshots. `code/configs/source_equivalence.json`
lists them. Build caches, upstream website documentation and promotional media
are excluded; experiment logic is not replaced or cosmetically renamed.

The packaged evaluator was run on the archived R3 Bonsai model: all three
quality metrics matched exactly. Recorded aggregation covers all 632 runs.
The earlier full local delivery was checked on nine models across eight host
environments. A fresh installation of every environment and a fresh training
of all experiments were not performed during packaging.

## Anonymization and Third-Party Code

The full supplementary archive's public logs have local account paths, host
names and personal identifiers redacted; untouched originals are retained
privately. Third-party copyright notices, licenses, scientific baseline names
and required dependency endpoints are retained. See
`code/THIRD_PARTY_NOTICES.md` before redistributing these implementations.
