#!/usr/bin/env python3
"""CUDA microbenchmark for the structural-objective autograd paths."""

import argparse
import json
import os
import statistics
import sys

import torch


def build_decisions():
    from factorgrad import FactorGradController

    controller = FactorGradController(
        enabled=True,
        topology_end_iteration=15_000,
        stabilization_steps=768,
        frozen_probability=2.0 / 3.0,
        seed=3407,
        schedule="full",
        early_start_iteration=3_000,
    )
    full_rgb = controller.decision(1_000)
    channel = controller.decision(3_000)
    l1_only = next(
        controller.decision(iteration)
        for iteration in range(15_768, 15_780)
        if controller.decision(iteration).mode == "l1_only"
    )
    return {
        "full_rgb": full_rgb,
        "factorized_channel": channel,
        "l1_only": l1_only,
    }


def benchmark(height, width, decision, lambda_dssim, warmup, repeats):
    from factorgrad import factorgrad_loss

    image = torch.rand((3, height, width), device="cuda", requires_grad=True)
    target = torch.rand_like(image)

    def step():
        image.grad = None
        result = factorgrad_loss(
            image,
            target,
            lambda_dssim,
            decision,
            importance_correction=True,
        )
        result.objective.backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def benchmark_structural(height, width, mode, warmup, repeats):
    from factorgrad.objective import (
        _ExactValueSelectedChannelSSIM,
        _fused,
    )

    image = torch.rand((1, 3, height, width), device="cuda", requires_grad=True)
    target = torch.rand_like(image)

    def step():
        image.grad = None
        if mode == "full_rgb":
            dssim = 1.0 - _fused(image, target, train=True)
            dssim.backward()
        elif mode == "factorized_channel":
            ssim = _ExactValueSelectedChannelSSIM.apply(
                image, target, 0, 1.0, True
            )
            (1.0 - ssim).backward()
        elif mode == "exact_value_only":
            with torch.no_grad():
                _ = 1.0 - _fused(image.detach(), target, train=False)
        else:
            raise ValueError("unknown structural benchmark mode: {}".format(mode))

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def benchmark_structural_backward(height, width, mode, warmup, repeats):
    """Time only the D-SSIM backward after its forward has reached the stream."""
    from factorgrad.objective import (
        _ExactValueSelectedChannelSSIM,
        _fused,
    )

    image = torch.rand((1, 3, height, width), device="cuda", requires_grad=True)
    target = torch.rand_like(image)

    def build_objective():
        image.grad = None
        if mode == "full_rgb":
            return 1.0 - _fused(image, target, train=True)
        if mode == "factorized_channel":
            return 1.0 - _ExactValueSelectedChannelSSIM.apply(
                image, target, 0, 1.0, True
            )
        raise ValueError("unknown structural backward mode: {}".format(mode))

    for _ in range(warmup):
        build_objective().backward()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        objective = build_objective()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        objective.backward()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    package_root = os.path.join(args.source_root, "factorgrad")
    sys.path.insert(0, package_root)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "note": "objective_paths time the training reconstruction objective with lambda_dssim=0.2; structural_paths directly time D-SSIM forward and its selected backward.",
        "sizes": {},
    }
    for height, width in ((533, 800), (1066, 1600)):
        size_key = "{}x{}".format(width, height)
        payload["sizes"][size_key] = {}
        for name, decision in build_decisions().items():
            payload["sizes"][size_key][name] = benchmark(
                height,
                width,
                decision,
                lambda_dssim=0.2,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        full_ms = payload["sizes"][size_key]["full_rgb"]["mean_ms"]
        for name in ("factorized_channel", "l1_only"):
            mode_ms = payload["sizes"][size_key][name]["mean_ms"]
            payload["sizes"][size_key][name]["reduction_vs_full_percent"] = (
                100.0 * (full_ms - mode_ms) / full_ms
            )
        structural_paths = {}
        for name in ("full_rgb", "factorized_channel", "exact_value_only"):
            structural_paths[name] = benchmark_structural(
                height,
                width,
                name,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        structural_full_ms = structural_paths["full_rgb"]["mean_ms"]
        for name in ("factorized_channel", "exact_value_only"):
            mode_ms = structural_paths[name]["mean_ms"]
            structural_paths[name]["reduction_vs_full_percent"] = (
                100.0 * (structural_full_ms - mode_ms) / structural_full_ms
            )
        payload["sizes"][size_key]["structural_paths"] = structural_paths

        structural_backward = {}
        for name in ("full_rgb", "factorized_channel"):
            structural_backward[name] = benchmark_structural_backward(
                height,
                width,
                name,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        backward_full_ms = structural_backward["full_rgb"]["mean_ms"]
        selected_ms = structural_backward["factorized_channel"]["mean_ms"]
        structural_backward["factorized_channel"][
            "reduction_vs_full_percent"
        ] = 100.0 * (backward_full_ms - selected_ms) / backward_full_ms
        payload["sizes"][size_key][
            "structural_backward_only"
        ] = structural_backward

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
