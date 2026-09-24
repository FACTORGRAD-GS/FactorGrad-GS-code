"""Comparable CUDA-event FPS benchmark for legacy 3DGS-style hosts."""

import json
import os
import sys
from argparse import ArgumentParser


SOURCE_ROOT = os.environ.get("FACTORGRAD_BENCHMARK_SOURCE")
if not SOURCE_ROOT:
    raise RuntimeError("FACTORGRAD_BENCHMARK_SOURCE is required")
sys.path.insert(0, os.path.abspath(SOURCE_ROOT))

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state


def _make_gaussians(sh_degree):
    mode = os.environ.get("FACTORGRAD_GAUSSIAN_MODE", "legacy")
    if mode == "taming":
        return GaussianModel(
            sh_degree,
            optimizer_type="default",
            rendering_mode="abs",
        )
    return GaussianModel(sh_degree)


@torch.no_grad()
def benchmark(dataset, iteration, pipeline, warmup, repeats, output):
    gaussians = _make_gaussians(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    views = scene.getTestCameras()
    if not views:
        raise RuntimeError("FPS benchmark requires test cameras")

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    def render_view(view):
        return render(view, gaussians, pipeline, background)["render"]

    for index in range(warmup):
        render_view(views[index % len(views)])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        for view in views:
            render_view(view)
    end.record()
    torch.cuda.synchronize()

    frames = repeats * len(views)
    elapsed_ms = start.elapsed_time(end)
    result = {
        "host": os.environ.get("FACTORGRAD_BENCHMARK_HOST", "legacy-3dgs"),
        "iteration": scene.loaded_iter,
        "test_views": len(views),
        "warmup_frames": warmup,
        "repeats": repeats,
        "timed_frames": frames,
        "elapsed_ms": elapsed_ms,
        "milliseconds_per_frame": elapsed_ms / frames,
        "fps": frames * 1000.0 / elapsed_ms,
        "gaussians": int(gaussians.get_xyz.shape[0]),
    }
    output = output or os.path.join(dataset.model_path, "fps_benchmark.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="Legacy 3DGS CUDA-event FPS benchmark")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    benchmark(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.warmup,
        args.repeats,
        args.output,
    )
