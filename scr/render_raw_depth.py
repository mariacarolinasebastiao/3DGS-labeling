#!/usr/bin/env python3
"""
render_raw_depth.py — Save RGB, float depth, and camera poses for every frame.

Usage:
    python render_raw_depth.py \
        --config /home/coder/data/zproject/outputs/video_processed/splatfacto/2026-05-09_193327/config.yml \
        --output-dir /home/coder/data/zproject/raw_renders
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2

from nerfstudio.utils.eval_utils import eval_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (output_dir / 'depth').mkdir(parents=True, exist_ok=True)
    (output_dir / 'poses').mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.config} ...")
    config, pipeline, _, _ = eval_setup(Path(args.config), test_mode='inference')
    model = pipeline.model
    device = pipeline.device

    train_cams  = pipeline.datamanager.train_dataset.cameras.to(device)
    eval_cams   = pipeline.datamanager.eval_dataset.cameras.to(device)

    train_paths = [p.name for p in pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames]
    eval_paths  = [p.name for p in pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames]

    all_cams = [(train_cams, train_paths, 'train'), (eval_cams, eval_paths, 'eval')]

    total = len(train_paths) + len(eval_paths)
    print(f"Rendering {total} cameras...")

    camera_params = {}
    idx = 0

    for cams, paths, split in all_cams:
        for i in range(len(cams)):
            idx += 1
            cam = cams[i:i+1]
            outputs = model.get_outputs_for_camera(cam)

            rgb   = outputs['rgb'].detach().cpu().numpy()
            depth = outputs['depth'].detach().cpu().numpy()
            depth = depth.squeeze(-1) if depth.ndim == 3 else depth

            stem = Path(paths[i]).stem

            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)# Save RGB
            cv2.imwrite(str(output_dir / 'rgb' / f'{stem}.png'), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
            
            np.save(output_dir / 'depth' / f'{stem}.npy', depth.astype(np.float32)) # Save depth

            # Save camera pose (c2w) as 4x4 matrix in nerfstudio training space
            c2w_34 = cam.camera_to_worlds[0].cpu().numpy()  # (3, 4)
            c2w_44 = np.eye(4, dtype=np.float64)
            c2w_44[:3, :] = c2w_34
            np.save(output_dir / 'poses' / f'{stem}.npy', c2w_44)

            # Save intrinsics (same for all frames in most cases, but save per-frame to be safe)
            camera_params[stem] = {'fx': float(cam.fx[0, 0].item()),
                'fy': float(cam.fy[0, 0].item()),
                'cx': float(cam.cx[0, 0].item()),
                'cy': float(cam.cy[0, 0].item()),
                'w':  int(cam.width[0, 0].item()),
                'h':  int(cam.height[0, 0].item()),}

            if idx % 10 == 0 or idx == total:
                print(f"  [{idx}/{total}] {stem}")

    with open(output_dir / 'camera_params.json', 'w') as f:
        json.dump(camera_params, f, indent=2)

    print(f"\nDone.")
    print(f"  RGB    → {output_dir/'rgb'}")
    print(f"  Depth  → {output_dir/'depth'}")
    print(f"  Poses  → {output_dir/'poses'}")
    print(f"  Params → {output_dir/'camera_params.json'}")


if __name__ == '__main__':
    main()