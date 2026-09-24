"""Measure test-view rendering FPS with warm-up and repeated CUDA-event sweeps."""

from __future__ import annotations

import itertools
import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import litegs
import litegs.config


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    lp_cdo, op_cdo, pp_cdo, dp_cdo = litegs.config.get_default_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo, parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo, parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo, parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo, parser)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(sys.argv[1:])

    lp = litegs.arguments.ModelParams.extract(args)
    op = litegs.arguments.OptimizationParams.extract(args)
    pp = litegs.arguments.PipelineParams.extract(args)

    cameras_info, camera_frames, _, _ = litegs.io_manager.load_colmap_result(
        lp.source_path, lp.images
    )
    for camera_frame in camera_frames:
        camera_frame.load_image()
    training_frames = [c for index, c in enumerate(camera_frames) if index % 8 != 0]
    test_frames = [c for index, c in enumerate(camera_frames) if index % 8 == 0]
    testset = litegs.data.CameraFrameDataset(
        cameras_info, test_frames, lp.resolution, pp.device_preload
    )
    test_batches = list(
        DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not pp.device_preload,
        )
    )
    if not test_batches:
        raise RuntimeError("test set is empty")

    xyz, scale, rot, sh_0, sh_rest, opacity = litegs.io_manager.load_ply(
        os.path.join(lp.model_path, "point_cloud", "finish", "point_cloud.ply"),
        lp.sh_degree,
    )
    # PLY readers return float64 NumPy arrays; the fused kernels require float32.
    xyz = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    scale = torch.tensor(scale, dtype=torch.float32, device="cuda")
    rot = torch.tensor(rot, dtype=torch.float32, device="cuda")
    sh_0 = torch.tensor(sh_0, dtype=torch.float32, device="cuda")
    sh_rest = torch.tensor(sh_rest, dtype=torch.float32, device="cuda")
    opacity = torch.tensor(opacity, dtype=torch.float32, device="cuda")
    cluster_origin = None
    cluster_extend = None
    if pp.cluster_size > 0:
        xyz, scale, rot, sh_0, sh_rest, opacity = (
            litegs.scene.point.spatial_refine(
                False, None, xyz, scale, rot, sh_0, sh_rest, opacity
            )
        )
        xyz, scale, rot, sh_0, sh_rest, opacity = litegs.scene.cluster.cluster_points(
            pp.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity
        )
        cluster_origin, cluster_extend = litegs.scene.cluster.get_cluster_AABB(
            xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
        )

    @torch.no_grad()
    def render_batch(batch) -> None:
        view_matrix, proj_matrix, frustumplane, gt_image, _ = batch
        view_matrix = view_matrix.cuda()
        proj_matrix = proj_matrix.cuda()
        frustumplane = frustumplane.cuda()
        (
            _,
            culled_xyz,
            culled_scale,
            culled_rot,
            culled_sh_0,
            culled_sh_rest,
            culled_opacity,
        ) = litegs.render.render_preprocess(
            cluster_origin,
            cluster_extend,
            frustumplane,
            xyz,
            scale,
            rot,
            sh_0,
            sh_rest,
            opacity,
            op,
            pp,
        )
        litegs.render.render(
            view_matrix,
            proj_matrix,
            culled_xyz,
            culled_scale,
            culled_rot,
            culled_sh_0,
            culled_sh_rest,
            culled_opacity,
            lp.sh_degree,
            gt_image.shape[2:],
            pp,
            False,
        )

    warmup_iterator = itertools.cycle(test_batches)
    for _ in range(args.warmup_frames):
        render_batch(next(warmup_iterator))
    torch.cuda.synchronize()

    elapsed_ms = 0.0
    for _ in range(args.repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for batch in test_batches:
            render_batch(batch)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms += start_event.elapsed_time(end_event)

    frame_count = len(test_batches) * args.repeats
    fps = frame_count * 1000.0 / elapsed_ms
    result = {
        "fps": fps,
        "test_views": len(test_batches),
        "warmup_frames": args.warmup_frames,
        "repeats": args.repeats,
        "elapsed_ms": elapsed_ms,
        "timing": "render_preprocess_plus_render_cuda_events",
    }
    output_path = os.path.join(lp.model_path, "fps_protocol.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
