"""CUDA-event inference FPS benchmark for the FastGS/SkipGS host."""

import json
import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_fastgs
from scene import Scene
from utils.general_utils import safe_state


@torch.no_grad()
def benchmark(dataset, iteration, pipeline, mult, warmup, repeats, output):
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
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
        return render_fastgs(view, gaussians, pipeline, background, mult)["render"]

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
        "host": "FastGS+SkipGS",
        "iteration": scene.loaded_iter,
        "test_views": len(views),
        "warmup_frames": warmup,
        "repeats": repeats,
        "timed_frames": frames,
        "elapsed_ms": elapsed_ms,
        "milliseconds_per_frame": elapsed_ms / frames,
        "fps": frames * 1000.0 / elapsed_ms,
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "mult": mult,
    }
    output = output or os.path.join(dataset.model_path, "fps_benchmark.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="FastGS CUDA-event FPS benchmark")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mult", default=0.5, type=float)
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
        args.mult,
        args.warmup,
        args.repeats,
        getattr(args, "output", None),
    )
