"""Non-invasive evidence recorder for FactorGrad-GS profiling runs."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import torch

from factorgrad import FactorDecision, factorgrad_loss


PROBE_ITERATIONS = {
    2500,
    3000,
    5000,
    10000,
    14500,
    15000,
    15008,
    15232,
    15552,
    15744,
    16000,
    20000,
    20032,
    25024,
    29952,
}


def training_phase(iteration, topology_end=15000, stabilization_steps=768):
    if iteration < 3000:
        return "warmup"
    if iteration < topology_end:
        return "active_densification"
    if iteration < topology_end + stabilization_steps:
        return "post_densification_stabilization"
    return "late_refinement"


def optimizer_will_step(iteration, final_iteration):
    if iteration >= final_iteration:
        return False
    if iteration <= 15000:
        return True
    if iteration <= 20000:
        return iteration % 32 == 0
    return iteration % 64 == 0


def _scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _norm(tensor):
    if tensor is None or tensor.numel() == 0:
        return None
    return _scalar(torch.linalg.vector_norm(tensor.detach().float()))


class P0EvidenceRecorder:
    EVENT_NAMES = (
        "render_start",
        "render_end",
        "loss_start",
        "loss_end",
        "backward_start",
        "image_grad_ready",
        "backward_end",
        "density_start",
        "density_end",
        "optimizer_start",
        "optimizer_end",
        "iteration_end",
    )

    def __init__(self, output_directory, enabled, topology_end, stabilization_steps):
        self.enabled = bool(enabled)
        self.output_directory = output_directory
        self.topology_end = int(topology_end)
        self.stabilization_steps = int(stabilization_steps)
        self.events = {}
        self.profile_totals = defaultdict(lambda: defaultdict(float))
        self.profile_counts = defaultdict(int)
        self.profile_file = None
        self.profile_writer = None
        self.gradient_path = os.path.join(output_directory, "p0_gradient_probe.jsonl")
        self.parameter_path = os.path.join(output_directory, "p0_parameter_trace.jsonl")
        self.topology_path = os.path.join(output_directory, "p0_topology_trace.jsonl")
        if not self.enabled:
            return
        os.makedirs(output_directory, exist_ok=True)
        self.events = {
            name: torch.cuda.Event(enable_timing=True) for name in self.EVENT_NAMES
        }
        profile_path = os.path.join(output_directory, "p0_profile_samples.csv")
        self.profile_file = open(profile_path, "w", newline="", encoding="utf-8")
        columns = [
            "iteration",
            "phase",
            "mode",
            "did_backward",
            "optimizer_step",
            "excluded_probe",
            "render_ms",
            "loss_forward_ms",
            "image_loss_backward_ms",
            "rasterizer_backward_ms",
            "density_control_ms",
            "optimizer_ms",
            "iteration_gpu_ms",
            "selector_cpu_us",
            "gate_cpu_us",
            "gaussians",
        ]
        self.profile_writer = csv.DictWriter(self.profile_file, fieldnames=columns)
        self.profile_writer.writeheader()

    def phase(self, iteration):
        return training_phase(
            iteration,
            topology_end=self.topology_end,
            stabilization_steps=self.stabilization_steps,
        )

    def should_probe(self, iteration):
        return self.enabled and int(iteration) in PROBE_ITERATIONS

    def record(self, name):
        if self.enabled:
            self.events[name].record()

    def image_gradient_hook(self, gradient):
        self.record("image_grad_ready")
        return gradient

    def _elapsed(self, start, end):
        return float(self.events[start].elapsed_time(self.events[end]))

    def record_profile(
        self,
        iteration,
        mode,
        did_backward,
        optimizer_step,
        selector_cpu_us,
        gate_cpu_us,
        gaussians,
    ):
        if not self.enabled:
            return
        phase = self.phase(iteration)
        excluded = self.should_probe(iteration)
        row = {
            "iteration": int(iteration),
            "phase": phase,
            "mode": str(mode),
            "did_backward": int(bool(did_backward)),
            "optimizer_step": int(bool(optimizer_step)),
            "excluded_probe": int(excluded),
            "render_ms": self._elapsed("render_start", "render_end"),
            "loss_forward_ms": self._elapsed("loss_start", "loss_end"),
            "image_loss_backward_ms": (
                self._elapsed("backward_start", "image_grad_ready")
                if did_backward
                else 0.0
            ),
            "rasterizer_backward_ms": (
                self._elapsed("image_grad_ready", "backward_end")
                if did_backward
                else 0.0
            ),
            "density_control_ms": self._elapsed("density_start", "density_end"),
            "optimizer_ms": self._elapsed("optimizer_start", "optimizer_end"),
            "iteration_gpu_ms": self._elapsed("render_start", "iteration_end"),
            "selector_cpu_us": float(selector_cpu_us),
            "gate_cpu_us": float(gate_cpu_us),
            "gaussians": int(gaussians),
        }
        self.profile_writer.writerow(row)
        if iteration % 100 == 0:
            self.profile_file.flush()
        if excluded:
            return
        key = "{}|{}|{}".format(
            phase, mode, "backward" if did_backward else "host_skipped"
        )
        self.profile_counts[key] += 1
        for field, value in row.items():
            if field.endswith("_ms") or field.endswith("_us"):
                self.profile_totals[key][field] += float(value)

    @staticmethod
    def _append_jsonl(path, payload):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def record_gradient_probe(self, iteration, image, target):
        if not self.should_probe(iteration):
            return
        phase = self.phase(iteration)
        temporal_probability = 2.0 / 3.0 if phase == "late_refinement" else 1.0

        def gradient_for(decision):
            probe = image.detach().clone().requires_grad_(True)
            result = factorgrad_loss(probe, target.detach(), 1.0, decision)
            gradient = torch.autograd.grad(result.objective, probe)[0].detach()
            return gradient, _scalar(result.exact_dssim)

        full_decision = FactorDecision(
            iteration=int(iteration),
            phase=phase,
            mode="full_rgb",
            build_structural_graph=True,
            selected_channel=None,
            temporal_probability=1.0,
            temporal_weight=1.0,
        )
        full_gradient, full_value = gradient_for(full_decision)
        full_norm_tensor = torch.linalg.vector_norm(full_gradient.float())
        denominator = torch.clamp(full_norm_tensor, min=1e-20)
        channel_gradients = []
        channel_rows = []
        for channel in range(3):
            decision = FactorDecision(
                iteration=int(iteration),
                phase=phase,
                mode="factorized_channel",
                build_structural_graph=True,
                selected_channel=channel,
                temporal_probability=temporal_probability,
                temporal_weight=1.0,
            )
            gradient, value = gradient_for(decision)
            channel_gradients.append(gradient)
            gradient_norm = torch.linalg.vector_norm(gradient.float())
            cosine = torch.sum(full_gradient.float() * gradient.float()) / (
                denominator * torch.clamp(gradient_norm, min=1e-20)
            )
            channel_rows.append(
                {
                    "channel": channel,
                    "cosine_to_full": _scalar(cosine),
                    "norm_ratio": _scalar(gradient_norm / denominator),
                    "exact_value_abs_error": abs(float(value) - float(full_value)),
                }
            )

        channel_mean = sum(channel_gradients) / 3.0
        relative_mean_error = torch.linalg.vector_norm(
            (channel_mean - full_gradient).float()
        ) / denominator
        channel_variance = sum(
            torch.sum((gradient.float() - full_gradient.float()) ** 2)
            for gradient in channel_gradients
        ) / 3.0
        full_energy = torch.clamp(torch.sum(full_gradient.float() ** 2), min=1e-20)

        temporal_variance = (1.0 - temporal_probability) * full_energy
        for gradient in channel_gradients:
            weighted = gradient.float() / temporal_probability
            temporal_variance = temporal_variance + (
                temporal_probability / 3.0
            ) * torch.sum((weighted - full_gradient.float()) ** 2)

        payload = {
            "iteration": int(iteration),
            "phase": phase,
            "image_height": int(image.shape[-2]),
            "image_width": int(image.shape[-1]),
            "full_dssim": float(full_value),
            "full_gradient_norm": _scalar(full_norm_tensor),
            "channel_mean_relative_error": _scalar(relative_mean_error),
            "channel_variance_ratio": _scalar(channel_variance / full_energy),
            "scheduled_temporal_probability": temporal_probability,
            "time_channel_variance_ratio": _scalar(temporal_variance / full_energy),
            "channels": channel_rows,
        }
        self._append_jsonl(self.gradient_path, payload)

    def record_parameter_state(self, iteration, gaussians, did_backward):
        if not self.should_probe(iteration):
            return
        for optimizer_name, optimizer in (
            ("main", gaussians.optimizer),
            ("sh", getattr(gaussians, "shoptimizer", None)),
        ):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                parameter = group["params"][0]
                state = optimizer.state.get(parameter, {})
                gradient = parameter.grad
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                param_rms = _scalar(torch.sqrt(torch.mean(parameter.detach().float() ** 2)))
                grad_rms = (
                    _scalar(torch.sqrt(torch.mean(gradient.detach().float() ** 2)))
                    if gradient is not None
                    else None
                )
                first_rms = (
                    _scalar(torch.sqrt(torch.mean(exp_avg.detach().float() ** 2)))
                    if exp_avg is not None
                    else None
                )
                second_root_mean = (
                    _scalar(torch.sqrt(torch.mean(exp_avg_sq.detach().float())))
                    if exp_avg_sq is not None
                    else None
                )
                moment_ratio = None
                if first_rms is not None and second_root_mean is not None:
                    moment_ratio = first_rms / max(second_root_mean, 1e-20)
                payload = {
                    "iteration": int(iteration),
                    "phase": self.phase(iteration),
                    "optimizer": optimizer_name,
                    "group": group.get("name", "unnamed"),
                    "did_backward": bool(did_backward),
                    "parameter_count": int(parameter.numel()),
                    "parameter_norm": _norm(parameter),
                    "parameter_rms": param_rms,
                    "gradient_norm": _norm(gradient),
                    "gradient_rms": grad_rms,
                    "first_moment_rms": first_rms,
                    "second_moment_root_mean": second_root_mean,
                    "first_to_second_moment_ratio": moment_ratio,
                }
                self._append_jsonl(self.parameter_path, payload)

    def record_topology(self, iteration, event, before, after, details=None):
        if not self.enabled:
            return
        payload = {
            "iteration": int(iteration),
            "phase": self.phase(iteration),
            "event": str(event),
            "before": int(before),
            "after": int(after),
            "net_change": int(after) - int(before),
        }
        if details:
            payload.update(details)
        self._append_jsonl(self.topology_path, payload)

    def close(self):
        if not self.enabled:
            return
        if self.profile_file is not None:
            self.profile_file.flush()
            self.profile_file.close()
        summary = {
            "probe_iterations_excluded": sorted(PROBE_ITERATIONS),
            "groups": {},
        }
        for key in sorted(self.profile_counts):
            count = self.profile_counts[key]
            group = {"count": count}
            for field, total in self.profile_totals[key].items():
                group["mean_{}".format(field)] = total / count
                group["sum_{}".format(field)] = total
            summary["groups"][key] = group
        with open(
            os.path.join(self.output_directory, "p0_phase_profile.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

