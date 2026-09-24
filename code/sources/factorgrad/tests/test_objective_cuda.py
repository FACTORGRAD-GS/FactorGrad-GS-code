import argparse
import json

import torch
from fused_ssim import fused_ssim

from factorgrad import FactorGradController, factorgrad_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.manual_seed(3407)
    image = torch.rand((3, 96, 128), device="cuda", requires_grad=True)
    target = torch.rand_like(image)

    baseline = 1.0 - fused_ssim(
        image.unsqueeze(0), target.unsqueeze(0), train=True
    )
    baseline.backward()
    baseline_gradient = image.grad.detach().clone()
    image.grad = None

    controller = FactorGradController(
        enabled=True,
        topology_end_iteration=0,
        stabilization_steps=0,
        frozen_probability=1.0,
        seed=3407,
    )
    per_channel = []
    exact_errors = []
    for iteration in range(3):
        decision = controller.decision(iteration)
        result = factorgrad_loss(image, target, 1.0, decision)
        exact_errors.append(abs(float(result.exact_dssim) - float(baseline.detach())))
        result.objective.backward()
        per_channel.append(image.grad.detach().clone())
        image.grad = None
        controller.record(decision, did_backward=True)

    averaged_gradient = sum(per_channel) / 3.0
    max_gradient_error = float((averaged_gradient - baseline_gradient).abs().max())

    time_only = FactorGradController(
        enabled=True,
        topology_end_iteration=0,
        stabilization_steps=0,
        frozen_probability=0.5,
        seed=3407,
        channel_sampling=False,
    )
    selected_decision = next(
        decision
        for decision in (time_only.decision(0), time_only.decision(1))
        if decision.mode == "reweighted_full_rgb"
    )
    time_only_result = factorgrad_loss(image, target, 1.0, selected_decision)
    time_only_exact_error = abs(
        float(time_only_result.exact_dssim) - float(baseline.detach())
    )
    time_only_result.objective.backward()
    time_only_gradient_error = float(
        (image.grad - 2.0 * baseline_gradient).abs().max()
    )
    image.grad = None

    report = {
        "max_exact_value_error": max(exact_errors),
        "max_cycle_mean_gradient_error": max_gradient_error,
        "time_only_exact_value_error": time_only_exact_error,
        "time_only_reweighted_gradient_error": time_only_gradient_error,
        "channels": [
            int(torch.count_nonzero(gradient.reshape(3, -1).abs().sum(1)).item())
            for gradient in per_channel
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    if report["max_exact_value_error"] > 1e-6:
        raise SystemExit("exact-value check failed")
    if report["max_cycle_mean_gradient_error"] > 2e-5:
        raise SystemExit("cycle-mean gradient check failed")
    if report["time_only_exact_value_error"] > 1e-6:
        raise SystemExit("time-only exact-value check failed")
    if report["time_only_reweighted_gradient_error"] > 2e-5:
        raise SystemExit("time-only reweighted-gradient check failed")


if __name__ == "__main__":
    main()
