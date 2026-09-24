# Result Files and Units

`results_ours.xlsx` contains FactorGrad integrations, ablations, sensitivity and
boundary runs. `baselines/results_baselines.xlsx` contains locally reproduced
unmodified comparison branches, including raw branches paired with transfers.
All 632 scene executions are retained in `per_run_per_scene.csv` and `.json`.
No best-run selection replaces the three-execution main-table averages.

Columns: PSNR in dB; SSIM and LPIPS dimensionless; process wall and native
training time in seconds; FPS in frames/second; peak memory in MiB; Gaussian
count as an integer. Native timer and wall timer have separate columns and
must not be substituted. Device peak and PyTorch allocated peak are different
measurements. `NR`/empty fields mean the original run did not provide that
measurement. Excel number formatting changes display only, not stored values.

`reference_tables/primary/` preserves the original dataset/run aggregations.
`reference_tables/mechanism/` contains CUDA microbenchmarks and gradient/profile
statistics. `evidence/` retains per-view results and configuration files so the
aggregations can be checked independently. Code and environment choices for
each run are recorded in `../code/configs/runs.json`.
