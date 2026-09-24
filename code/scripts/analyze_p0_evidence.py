#!/usr/bin/env python3
"""Aggregate the P0 profiling, gradient, Adam, topology, and quality evidence."""

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict, deque


METHODS = ("skipgs", "factorgrad_full")
SCENES = ("bicycle", "kitchen", "train", "drjohnson")
PHASE_ORDER = (
    "warmup",
    "active_densification",
    "post_densification_stabilization",
    "late_refinement",
)
PROFILE_COMPONENTS = (
    "render_ms",
    "loss_forward_ms",
    "image_loss_backward_ms",
    "rasterizer_backward_ms",
    "density_control_ms",
    "optimizer_ms",
)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(finite) if finite else math.nan


def std(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(archive):
    runs = {}
    for method in METHODS:
        for scene in SCENES:
            root = os.path.join(archive, "runs", method, scene)
            required = (
                "p0_profile_samples.csv",
                "p0_gradient_probe.jsonl",
                "p0_parameter_trace.jsonl",
                "p0_topology_trace.jsonl",
                "results.json",
                "runtime_summary.json",
            )
            missing = [name for name in required if not os.path.exists(os.path.join(root, name))]
            if missing:
                raise FileNotFoundError("{} missing {}".format(root, ", ".join(missing)))
            runs[(method, scene)] = {
                "root": root,
                "profile": read_csv(os.path.join(root, "p0_profile_samples.csv")),
                "gradient": read_jsonl(os.path.join(root, "p0_gradient_probe.jsonl")),
                "parameter": read_jsonl(os.path.join(root, "p0_parameter_trace.jsonl")),
                "topology": read_jsonl(os.path.join(root, "p0_topology_trace.jsonl")),
                "results": read_json(os.path.join(root, "results.json"))["ours_30000"],
                "runtime": read_json(os.path.join(root, "runtime_summary.json")),
            }
    return runs


def summarize_profiles(runs):
    grouped = defaultdict(list)
    mode_grouped = defaultdict(list)
    component_totals = defaultdict(lambda: defaultdict(float))
    scene_totals = []
    for (method, scene), run in runs.items():
        retained = [row for row in run["profile"] if int(row["excluded_probe"]) == 0]
        for row in retained:
            grouped[(method, row["phase"])].append(row)
            mode_grouped[
                (method, row["phase"], row["mode"], int(row["did_backward"]))
            ].append(row)
            for component in PROFILE_COMPONENTS:
                component_totals[method][component] += float(row[component])
        scene_totals.append(
            {
                "method": method,
                "scene": scene,
                "profiled_iterations": len(retained),
                "iteration_gpu_seconds": sum(float(row["iteration_gpu_ms"]) for row in retained) / 1000.0,
                "training_time_seconds": float(run["runtime"]["training_time_seconds"]),
            }
        )

    rows = []
    for method in METHODS:
        for phase in PHASE_ORDER:
            samples = grouped[(method, phase)]
            if not samples:
                continue
            row = {
                "method": method,
                "phase": phase,
                "iterations": len(samples),
                "backward_ratio": mean(float(item["did_backward"]) for item in samples),
                "optimizer_step_ratio": mean(float(item["optimizer_step"]) for item in samples),
                "selector_cpu_us": mean(float(item["selector_cpu_us"]) for item in samples),
                "gate_cpu_us": mean(float(item["gate_cpu_us"]) for item in samples),
            }
            for component in PROFILE_COMPONENTS + ("iteration_gpu_ms",):
                row["mean_" + component] = mean(float(item[component]) for item in samples)
                row["sum_" + component] = sum(float(item[component]) for item in samples)
            rows.append(row)
    mode_rows = []
    for key in sorted(mode_grouped):
        method, phase, mode, did_backward = key
        samples = mode_grouped[key]
        row = {
            "method": method,
            "phase": phase,
            "mode": mode,
            "did_backward": did_backward,
            "iterations": len(samples),
            "selector_cpu_us": mean(float(item["selector_cpu_us"]) for item in samples),
            "gate_cpu_us": mean(float(item["gate_cpu_us"]) for item in samples),
        }
        for component in PROFILE_COMPONENTS + ("iteration_gpu_ms",):
            row["mean_" + component] = mean(
                float(item[component]) for item in samples
            )
            row["sum_" + component] = sum(
                float(item[component]) for item in samples
            )
        mode_rows.append(row)
    return rows, mode_rows, component_totals, scene_totals


def summarize_gradients(runs):
    grouped = defaultdict(list)
    for (method, scene), run in runs.items():
        for sample in run["gradient"]:
            sample = dict(sample)
            sample["scene"] = scene
            grouped[(method, sample["phase"])].append(sample)

    rows = []
    for method in METHODS:
        for phase in PHASE_ORDER:
            samples = grouped[(method, phase)]
            if not samples:
                continue
            cosines = [channel["cosine_to_full"] for item in samples for channel in item["channels"]]
            norm_ratios = [channel["norm_ratio"] for item in samples for channel in item["channels"]]
            exact_errors = [channel["exact_value_abs_error"] for item in samples for channel in item["channels"]]
            rows.append(
                {
                    "method": method,
                    "phase": phase,
                    "probes": len(samples),
                    "mean_channel_cosine": mean(cosines),
                    "std_channel_cosine": std(cosines),
                    "mean_channel_norm_ratio": mean(norm_ratios),
                    "std_channel_norm_ratio": std(norm_ratios),
                    "mean_channel_variance_ratio": mean(item["channel_variance_ratio"] for item in samples),
                    "mean_time_channel_variance_ratio": mean(item["time_channel_variance_ratio"] for item in samples),
                    "max_cycle_relative_error": max(item["channel_mean_relative_error"] for item in samples),
                    "max_exact_value_abs_error": max(exact_errors),
                }
            )
    return rows


def summarize_parameters(runs):
    grouped = defaultdict(list)
    for (method, scene), run in runs.items():
        for sample in run["parameter"]:
            grouped[(method, sample["phase"], sample["optimizer"], sample["group"])].append(sample)
    rows = []
    for key in sorted(grouped):
        method, phase, optimizer, group = key
        samples = grouped[key]
        rows.append(
            {
                "method": method,
                "phase": phase,
                "optimizer": optimizer,
                "group": group,
                "samples": len(samples),
                "gradient_norm_mean": mean(item["gradient_norm"] for item in samples),
                "gradient_norm_std": std(item["gradient_norm"] for item in samples),
                "gradient_rms_mean": mean(item["gradient_rms"] for item in samples),
                "first_moment_rms_mean": mean(item["first_moment_rms"] for item in samples),
                "second_moment_root_mean": mean(item["second_moment_root_mean"] for item in samples),
                "first_to_second_moment_ratio_mean": mean(item["first_to_second_moment_ratio"] for item in samples),
            }
        )
    return rows


def paired_parameter_ratios(parameter_rows):
    indexed = {
        (row["method"], row["phase"], row["optimizer"], row["group"]): row
        for row in parameter_rows
    }
    rows = []
    keys = sorted(
        {
            (row["phase"], row["optimizer"], row["group"])
            for row in parameter_rows
        }
    )
    fields = (
        "gradient_rms_mean",
        "first_moment_rms_mean",
        "second_moment_root_mean",
        "first_to_second_moment_ratio_mean",
    )
    for phase, optimizer, group in keys:
        base = indexed.get(("skipgs", phase, optimizer, group))
        ours = indexed.get(("factorgrad_full", phase, optimizer, group))
        if base is None or ours is None:
            continue
        row = {"phase": phase, "optimizer": optimizer, "group": group}
        for field in fields:
            denominator = float(base[field])
            row[field.replace("_mean", "_ratio")] = (
                float(ours[field]) / denominator
                if math.isfinite(denominator) and abs(denominator) > 1e-20
                else math.nan
            )
        rows.append(row)
    return rows


def summarize_topology(runs):
    rows = []
    for (method, scene), run in runs.items():
        totals = defaultdict(int)
        for event in run["topology"]:
            for key in (
                "net_change",
                "clone_parents",
                "split_parents",
                "split_children",
                "prune_candidates",
                "prune_applied",
            ):
                totals[key] += int(event.get(key, 0))
        profile = run["profile"]
        rows.append(
            {
                "method": method,
                "scene": scene,
                "initial_profiled_gaussians": int(float(profile[0]["gaussians"])),
                "final_gaussians": int(run["runtime"]["gaussians"]),
                "topology_events": len(run["topology"]),
                **dict(totals),
            }
        )
    return rows


def paired_topology_differences(topology_rows):
    indexed = {(row["method"], row["scene"]): row for row in topology_rows}
    rows = []
    for scene in SCENES:
        base = indexed[("skipgs", scene)]
        ours = indexed[("factorgrad_full", scene)]
        row = {"scene": scene}
        for field in (
            "final_gaussians",
            "clone_parents",
            "split_parents",
            "split_children",
            "prune_candidates",
            "prune_applied",
        ):
            row["skipgs_" + field] = int(base.get(field, 0))
            row["factorgrad_" + field] = int(ours.get(field, 0))
            row["delta_" + field] = int(ours.get(field, 0)) - int(
                base.get(field, 0)
            )
        rows.append(row)
    return rows


def summarize_quality(runs):
    rows = []
    for scene in SCENES:
        base = runs[("skipgs", scene)]
        ours = runs[("factorgrad_full", scene)]
        row = {"scene": scene}
        for metric in ("PSNR", "SSIM", "LPIPS"):
            row["skipgs_" + metric.lower()] = float(base["results"][metric])
            row["factorgrad_" + metric.lower()] = float(ours["results"][metric])
            row["delta_" + metric.lower()] = float(ours["results"][metric]) - float(base["results"][metric])
        base_time = float(base["runtime"]["training_time_seconds"])
        ours_time = float(ours["runtime"]["training_time_seconds"])
        row["skipgs_time_seconds"] = base_time
        row["factorgrad_time_seconds"] = ours_time
        row["speedup_percent"] = 100.0 * (base_time - ours_time) / base_time
        rows.append(row)
    return rows


def load_tensorboard_scalar(root, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(root, size_guidance={"scalars": 0})
    accumulator.Reload()
    return [(int(item.step), float(item.value)) for item in accumulator.Scalars(tag)]


def summarize_loss_trajectories(runs):
    phase_bounds = (
        ("warmup", 1, 2999),
        ("active_densification", 3000, 14999),
        ("post_densification_stabilization", 15000, 15767),
        ("late_refinement", 15768, 30000),
    )
    trajectories = {}
    rows = []
    for (method, scene), run in runs.items():
        series = load_tensorboard_scalar(
            run["root"], "train_loss_patches/total_loss"
        )
        trajectories[(method, scene)] = series
        for phase, start, end in phase_bounds:
            values = [value for step, value in series if start <= step <= end]
            rows.append(
                {
                    "method": method,
                    "scene": scene,
                    "phase": phase,
                    "samples": len(values),
                    "loss_mean": mean(values),
                    "loss_std": std(values),
                }
            )
    return trajectories, rows


def rolling_mean(series, window=256):
    queue = deque()
    running = 0.0
    output = []
    for step, value in series:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.popleft()
        output.append((step, running / len(queue)))
    return output


def make_figure(archive, runs, component_totals, gradient_rows):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0))

    labels = [item.replace("_ms", "").replace("_", "\n") for item in PROFILE_COMPONENTS]
    x = list(range(len(labels)))
    width = 0.36
    for offset, method, title in ((-width / 2, "skipgs", "SkipGS"), (width / 2, "factorgrad_full", "FactorGrad-GS")):
        values = [component_totals[method][name] / 1000.0 for name in PROFILE_COMPONENTS]
        axes[0].bar([value + offset for value in x], values, width=width, label=title)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Cumulative GPU time (s)")
    axes[0].set_title("(a) Training cost by component")
    axes[0].legend(frameon=False)

    phase_rows = {row["phase"]: row for row in gradient_rows if row["method"] == "factorgrad_full"}
    phases = [phase for phase in PHASE_ORDER if phase in phase_rows]
    applied_variance = {
        "warmup": 0.0,
        "active_densification": phase_rows["active_densification"]["mean_channel_variance_ratio"],
        "post_densification_stabilization": 0.0,
        "late_refinement": phase_rows["late_refinement"]["mean_time_channel_variance_ratio"],
    }
    axes[1].bar(
        range(len(phases)),
        [applied_variance[phase] for phase in phases],
        color="#4C78A8",
    )
    axes[1].set_xticks(range(len(phases)), ["warm", "densify", "recover", "late"][: len(phases)])
    axes[1].set_ylabel("Variance / full-gradient energy")
    axes[1].set_title("(b) Applied estimator variance")

    colors = {"bicycle": "#4C78A8", "kitchen": "#F58518", "train": "#54A24B", "drjohnson": "#B279A2"}
    for scene in SCENES:
        for method, linestyle in (("skipgs", "--"), ("factorgrad_full", "-")):
            profile = runs[(method, scene)]["profile"]
            points = profile[::100]
            axes[2].plot(
                [int(row["iteration"]) / 1000.0 for row in points],
                [float(row["gaussians"]) / 1e6 for row in points],
                color=colors[scene],
                linestyle=linestyle,
                linewidth=1.0,
                label="{} {}".format(scene, "FG" if method == "factorgrad_full" else "Skip"),
            )
    axes[2].set_xlabel("Iteration (K)")
    axes[2].set_ylabel("Gaussians (M)")
    axes[2].set_title("(c) Topology trajectories")
    axes[2].legend(frameon=False, fontsize=6, ncol=2)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    for extension in ("pdf", "svg"):
        fig.savefig(os.path.join(archive, "results", "p0_mechanism_evidence." + extension), bbox_inches="tight")
    plt.close(fig)


def make_loss_figure(archive, trajectories):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    titles = {
        "bicycle": "Bicycle",
        "kitchen": "Kitchen",
        "train": "Train",
        "drjohnson": "Dr Johnson",
    }
    for axis, scene in zip(axes.flat, SCENES):
        for method, color, label in (
            ("skipgs", "#4C78A8", "SkipGS"),
            ("factorgrad_full", "#E07B54", "FactorGrad-GS"),
        ):
            smoothed = rolling_mean(trajectories[(method, scene)])
            axis.plot(
                [step / 1000.0 for step, _ in smoothed],
                [value for _, value in smoothed],
                color=color,
                linewidth=1.0,
                label=label,
            )
        axis.axvline(15.0, color="#666666", linewidth=0.6, linestyle="--")
        axis.axvline(15.768, color="#999999", linewidth=0.6, linestyle=":")
        axis.set_title(titles[scene])
        axis.set_xlabel("Iteration (K)")
        axis.set_ylabel("Training objective")
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.tight_layout()
    for extension in ("pdf", "svg"):
        fig.savefig(
            os.path.join(
                archive, "results", "p0_training_objective_trajectories." + extension
            ),
            bbox_inches="tight",
        )
    plt.close(fig)


def write_summary(
    archive,
    profile_rows,
    mode_rows,
    component_totals,
    gradient_rows,
    parameter_ratio_rows,
    topology_difference_rows,
    loss_rows,
    quality_rows,
):
    micro = read_json(os.path.join(archive, "results", "structural_objective_microbenchmark.json"))
    path = os.path.join(archive, "results", "summary.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# FactorGrad-GS P0 mechanism evidence\n\n")
        handle.write("All values below come from the four-scene audit campaign. Probe iterations are excluded from timing aggregates.\n\n")
        handle.write("## Structural-objective microbenchmark\n\n")
        handle.write("| Resolution | Full RGB (ms) | One channel (ms) | L1-only (ms) | One-channel reduction |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for size in ("800x533", "1600x1066"):
            item = micro["sizes"][size]
            handle.write(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f}% |\n".format(
                    size,
                    item["full_rgb"]["mean_ms"],
                    item["factorized_channel"]["mean_ms"],
                    item["l1_only"]["mean_ms"],
                    item["factorized_channel"]["reduction_vs_full_percent"],
                )
            )

        handle.write("\n### Direct D-SSIM path\n\n")
        handle.write("| Resolution | Full RGB (ms) | One channel (ms) | Exact value only (ms) | One-channel reduction |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for size in ("800x533", "1600x1066"):
            item = micro["sizes"][size]["structural_paths"]
            handle.write(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f}% |\n".format(
                    size,
                    item["full_rgb"]["mean_ms"],
                    item["factorized_channel"]["mean_ms"],
                    item["exact_value_only"]["mean_ms"],
                    item["factorized_channel"]["reduction_vs_full_percent"],
                )
            )

        if all(
            "structural_backward_only" in micro["sizes"][size]
            for size in ("800x533", "1600x1066")
        ):
            handle.write("\n### D-SSIM backward only\n\n")
            handle.write("| Resolution | Full RGB (ms) | One channel (ms) | Reduction |\n")
            handle.write("|---|---:|---:|---:|\n")
            for size in ("800x533", "1600x1066"):
                item = micro["sizes"][size]["structural_backward_only"]
                handle.write(
                    "| {} | {:.4f} | {:.4f} | {:.2f}% |\n".format(
                        size,
                        item["full_rgb"]["mean_ms"],
                        item["factorized_channel"]["mean_ms"],
                        item["factorized_channel"]["reduction_vs_full_percent"],
                    )
                )

        handle.write("\n## Cumulative GPU profile\n\n")
        handle.write("| Method | Render | Loss forward | Loss backward | Raster backward | Density | Optimizer |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for method in METHODS:
            total = component_totals[method]
            handle.write(
                "| {} | {:.2f}s | {:.2f}s | {:.2f}s | {:.2f}s | {:.2f}s | {:.2f}s |\n".format(
                    method,
                    total["render_ms"] / 1000.0,
                    total["loss_forward_ms"] / 1000.0,
                    total["image_loss_backward_ms"] / 1000.0,
                    total["rasterizer_backward_ms"] / 1000.0,
                    total["density_control_ms"] / 1000.0,
                    total["optimizer_ms"] / 1000.0,
                )
            )

        handle.write("\n## Backward path microprofile inside training\n\n")
        handle.write("| Method | Phase | Path | Backward? | Iterations | Loss backward (ms/iter) | Raster backward (ms/iter) | Iteration GPU (ms) |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in mode_rows:
            handle.write(
                "| {method} | {phase} | {mode} | {did_backward} | {iterations} | {mean_image_loss_backward_ms:.4f} | {mean_rasterizer_backward_ms:.4f} | {mean_iteration_gpu_ms:.4f} |\n".format(
                    **row
                )
            )

        handle.write("\n## Final paired quality\n\n")
        handle.write("| Scene | dPSNR | dSSIM | dLPIPS | Internal speedup |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in quality_rows:
            handle.write(
                "| {scene} | {delta_psnr:+.4f} | {delta_ssim:+.5f} | {delta_lpips:+.5f} | {speedup_percent:+.2f}% |\n".format(**row)
            )

        handle.write("\n## Estimator checks\n\n")
        handle.write("| Method | Phase | Channel cosine | Channel variance ratio | Time-channel variance ratio | Max cycle error |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for row in gradient_rows:
            handle.write(
                "| {method} | {phase} | {mean_channel_cosine:.4f} | {mean_channel_variance_ratio:.4f} | {mean_time_channel_variance_ratio:.4f} | {max_cycle_relative_error:.3e} |\n".format(**row)
            )

        handle.write("\n## Adam-state ratios: FactorGrad-GS / SkipGS\n\n")
        handle.write("| Phase | Optimizer | Group | Gradient RMS | First moment RMS | Second moment root mean | Moment ratio |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|\n")
        for row in parameter_ratio_rows:
            handle.write(
                "| {phase} | {optimizer} | {group} | {gradient_rms_ratio:.4f} | {first_moment_rms_ratio:.4f} | {second_moment_root_ratio:.4f} | {first_to_second_moment_ratio_ratio:.4f} |\n".format(
                    **row
                )
            )

        handle.write("\n## Paired topology differences\n\n")
        handle.write("| Scene | Final Gaussian delta | Clone-parent delta | Split-parent delta | Pruned delta |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in topology_difference_rows:
            handle.write(
                "| {scene} | {delta_final_gaussians:+d} | {delta_clone_parents:+d} | {delta_split_parents:+d} | {delta_prune_applied:+d} |\n".format(
                    **row
                )
            )

        handle.write("\n## Training-objective trajectories\n\n")
        handle.write("| Method | Scene | Phase | Mean objective | Std. |\n")
        handle.write("|---|---|---|---:|---:|\n")
        for row in loss_rows:
            handle.write(
                "| {method} | {scene} | {phase} | {loss_mean:.6f} | {loss_std:.6f} |\n".format(
                    **row
                )
            )

        handle.write("\nRaw per-iteration, per-probe, parameter-group, and topology-event records remain in each run directory.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    results_dir = os.path.join(args.archive, "results")
    os.makedirs(results_dir, exist_ok=True)

    runs = load_runs(args.archive)
    profile_rows, mode_rows, component_totals, scene_totals = summarize_profiles(runs)
    gradient_rows = summarize_gradients(runs)
    parameter_rows = summarize_parameters(runs)
    parameter_ratio_rows = paired_parameter_ratios(parameter_rows)
    topology_rows = summarize_topology(runs)
    topology_difference_rows = paired_topology_differences(topology_rows)
    trajectories, loss_rows = summarize_loss_trajectories(runs)
    quality_rows = summarize_quality(runs)

    write_csv(os.path.join(results_dir, "phase_profile.csv"), list(profile_rows[0]), profile_rows)
    write_csv(os.path.join(results_dir, "mode_profile.csv"), list(mode_rows[0]), mode_rows)
    write_csv(os.path.join(results_dir, "profile_scene_totals.csv"), list(scene_totals[0]), scene_totals)
    write_csv(os.path.join(results_dir, "gradient_summary.csv"), list(gradient_rows[0]), gradient_rows)
    write_csv(os.path.join(results_dir, "parameter_adam_summary.csv"), list(parameter_rows[0]), parameter_rows)
    write_csv(os.path.join(results_dir, "parameter_adam_paired_ratios.csv"), list(parameter_ratio_rows[0]), parameter_ratio_rows)
    write_csv(os.path.join(results_dir, "topology_summary.csv"), list(topology_rows[0]), topology_rows)
    write_csv(os.path.join(results_dir, "topology_paired_differences.csv"), list(topology_difference_rows[0]), topology_difference_rows)
    write_csv(os.path.join(results_dir, "loss_phase_summary.csv"), list(loss_rows[0]), loss_rows)
    write_csv(os.path.join(results_dir, "paired_quality.csv"), list(quality_rows[0]), quality_rows)
    make_figure(args.archive, runs, component_totals, gradient_rows)
    make_loss_figure(args.archive, trajectories)
    write_summary(
        args.archive,
        profile_rows,
        mode_rows,
        component_totals,
        gradient_rows,
        parameter_ratio_rows,
        topology_difference_rows,
        loss_rows,
        quality_rows,
    )


if __name__ == "__main__":
    main()
