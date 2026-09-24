# Third-Party Code and Attribution

The retained host implementations include 3DGS, FastGS, Speedy-Splat,
Taming-3DGS, DashGaussian, LeGS, Shorter-GS/LiteGS, their CUDA dependencies,
and metric implementations. Their license files, copyright headers, author
contacts and upstream references have been retained.
`THIRD_PARTY_LICENSE_FILES.json` indexes the retained license documents.

SkipGS and Mini-Splatting source are not redistributed in this anonymous
repository. Reviewers who obtain either dependency separately must retain its
original license and attribution. Recorded numeric results for these methods
remain in `results/`.

## DashGaussian

The DashGaussian code in `sources/baselines/dash/` and
`sources/transfer/dash/` comes from the [official DashGaussian repository](https://github.com/YouyuChen0207/DashGaussian).
Its upstream README credits the DashGaussian paper to Youyu Chen, Junjun Jiang,
Kui Jiang, Xiao Tang, Zhihao Li, Xianming Liu and Yinyu Nie. Both bundled
folders retain the upstream `LICENSE.md` text, which states Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). The upstream README
also displays a CC BY-NC-SA 4.0 badge; that badge and the license text disagree.
This notice records the discrepancy without replacing the upstream license.

Compared with upstream commit `4e3b560`, the bundled `train_dash.py` adds
FactorGrad controller and loss integration, command-line options, and training
statistics; `benchmark_fps.py` is an added benchmark helper. The two bundled
DashGaussian folders contain identical files. For noncommercial sharing,
retain the upstream credit, this modification notice, and the applicable
license text. Commercial rights are not granted by the bundled CC BY-NC 4.0
text.

FactorGrad's controller and objective implement the paper's contribution;
the host rasterizer, baseline gate and third-party metric implementations
are not claimed as new FactorGrad code. Host class and CUDA symbol names were
not cosmetically renamed: doing so would obscure provenance and risk breaking
checkpoint/extension compatibility. The FactorGrad-specific names already
distinguish the contribution (`FactorGradController`, `factorgrad_loss`).

No new blanket license is applied to the assembled repository, and this notice
grants no additional license for the original FactorGrad contribution at this
time. Individual third-party licenses continue to apply, including research-use
restrictions where present. Review the bundled components' redistribution
requirements before sharing them. This folder is prepared for source handoff,
not proof that every component permits every form of commercial redistribution.

Original source snapshots are kept apart from new portable entry scripts.
Unused local launchers with private absolute paths were moved outside the
public repository. Required download links and third-party attribution remain.
Authors should review all retained attribution against their double-blind
submission policy before publication.
