"""CUDA-event FPS benchmark for the official ShorterSplatting implementation."""

import json
import os
import sys
from argparse import ArgumentParser


SOURCE_ROOT = os.environ.get("FACTORGRAD_BENCHMARK_SOURCE")
if not SOURCE_ROOT:
    raise RuntimeError("FACTORGRAD_BENCHMARK_SOURCE is required")
sys.path.insert(0, os.path.abspath(SOURCE_ROOT))

import torch
from torch.utils.data import DataLoader

import litegs
import litegs.config


def main():
    parser = ArgumentParser(description="ShorterSplatting CUDA-event FPS benchmark")
    lp_default, op_default, pp_default, dp_default = litegs.config.get_default_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_default, parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_default, parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_default, parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_default, parser)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    lp = litegs.arguments.ModelParams.extract(args)
    op = litegs.arguments.OptimizationParams.extract(args)
    pp = litegs.arguments.PipelineParams.extract(args)

    camera_info, camera_frames, _, _ = litegs.io_manager.load_colmap_result(
        lp.source_path,
        lp.images,
    )
    for camera_frame in camera_frames:
        camera_frame.load_image()
    test_frames = [frame for index, frame in enumerate(camera_frames) if index % 8 == 0]
    test_set = litegs.data.CameraFrameDataset(
        camera_info,
        test_frames,
        lp.resolution,
        pp.device_preload,
    )
    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=not pp.device_preload,
    )
    batches = list(loader)
    if not batches:
        raise RuntimeError("FPS benchmark requires test cameras")

    xyz, scale, rot, sh_0, sh_rest, opacity = litegs.io_manager.load_ply(
        os.path.join(lp.model_path, "point_cloud", "finish", "point_cloud.ply"),
        lp.sh_degree,
    )
    xyz = torch.as_tensor(xyz, dtype=torch.float32, device="cuda")
    scale = torch.as_tensor(scale, dtype=torch.float32, device="cuda")
    rot = torch.as_tensor(rot, dtype=torch.float32, device="cuda")
    sh_0 = torch.as_tensor(sh_0, dtype=torch.float32, device="cuda")
    sh_rest = torch.as_tensor(sh_rest, dtype=torch.float32, device="cuda")
    opacity = torch.as_tensor(opacity, dtype=torch.float32, device="cuda")
    cluster_origin = None
    cluster_extend = None
    if pp.cluster_size > 0:
        xyz, scale, rot, sh_0, sh_rest, opacity = litegs.scene.point.spatial_refine(
            False,
            None,
            xyz,
            scale,
            rot,
            sh_0,
            sh_rest,
            opacity,
        )
        xyz, scale, rot, sh_0, sh_rest, opacity = litegs.scene.cluster.cluster_points(
            pp.cluster_size,
            xyz,
            scale,
            rot,
            sh_0,
            sh_rest,
            opacity,
        )
        cluster_origin, cluster_extend = litegs.scene.cluster.get_cluster_AABB(
            xyz,
            scale.exp(),
            torch.nn.functional.normalize(rot, dim=0),
        )

    def render_batch(batch):
        view_matrix, proj_matrix, frustum_plane, gt_image, _ = batch
        view_matrix = view_matrix.cuda()
        proj_matrix = proj_matrix.cuda()
        frustum_plane = frustum_plane.cuda()
        _, c_xyz, c_scale, c_rot, c_sh0, c_shrest, c_opacity = (
            litegs.render.render_preprocess(
                cluster_origin,
                cluster_extend,
                frustum_plane,
                xyz,
                scale,
                rot,
                sh_0,
                sh_rest,
                opacity,
                op,
                pp,
            )
        )
        return litegs.render.render(
            view_matrix,
            proj_matrix,
            c_xyz,
            c_scale,
            c_rot,
            c_sh0,
            c_shrest,
            c_opacity,
            lp.sh_degree,
            gt_image.shape[2:],
            pp,
            False,
        )[0]

    with torch.no_grad():
        for index in range(args.warmup):
            render_batch(batches[index % len(batches)])
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            for batch in batches:
                render_batch(batch)
        end.record()
        torch.cuda.synchronize()

    frames = args.repeats * len(batches)
    elapsed_ms = start.elapsed_time(end)
    result = {
        "host": "ShorterSplatting",
        "test_views": len(batches),
        "warmup_frames": args.warmup,
        "repeats": args.repeats,
        "timed_frames": frames,
        "elapsed_ms": elapsed_ms,
        "milliseconds_per_frame": elapsed_ms / frames,
        "fps": frames * 1000.0 / elapsed_ms,
        "gaussians": int(xyz.shape[-2] if xyz.ndim > 2 else xyz.shape[0]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
