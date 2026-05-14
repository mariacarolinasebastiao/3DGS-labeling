#!/usr/bin/env python3
"""
label_gaussians.py — Depth-based semantic labeling of Gaussian splats
                     using YOLO-World (open-vocabulary detection) + SAM2 (precise masks).

Usage:
    python label_gaussians.py \
        --renders-dir /home/coder/data/zproject/raw_renders \
        --splat-ply   /home/coder/data/zproject/exports/splat1/splat.ply \
        --output-dir  /home/coder/data/zproject/output
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import cv2
from scipy.spatial import cKDTree
from ultralytics import YOLO, SAM
from plyfile import PlyData, PlyElement


CLASSES = ['bed', 'wardrobe', 'desk', 'dining table', 'coffee table','kitchen counter', 'cabinet',
    'chair', 'armchair', 'couch',    'laptop', 'monitor', 'tv', 'keyboard', 'mouse','sink', 'microwave', 'toaster', 'kettle', 'coffee machine',
    'lamp', 'notebook', 'bottle', 'cup', 'plate','picture frame', 'mirror','backpack', 'suitcase', 'pillow', 'blanket',    'shower', 'bathtub',]

LABEL_COLORS = {
    'bed':              [0,   0,   255],
    'wardrobe':         [128, 0,   128],
    'desk':             [255, 0,   200],
    'dining table':     [255, 165, 0  ],
    'coffee table':     [200, 100, 50 ],
    'kitchen counter':  [160, 100, 80 ],
    'cabinet':          [140, 80,  60 ],
    'chair':            [0,   200, 0  ],
    'armchair':         [0,   180, 80 ],
    'couch':            [0,   128, 255],
    'laptop':           [255, 255, 0  ],
    'monitor':          [200, 200, 100],
    'tv':               [64,  64,  255],
    'keyboard':         [100, 200, 200],
    'mouse':            [200, 100, 200],
    'sink':             [0,   255, 200],
    'microwave':        [255, 128, 0  ],
    'toaster':          [255, 80,  80 ],
    'kettle':           [220, 150, 60 ],
    'coffee machine':   [180, 100, 50 ],
    'lamp':             [255, 230, 100],
    'notebook':         [200, 200, 0  ],
    'bottle':           [200, 100, 50 ],
    'cup':              [200, 150, 50 ],
    'plate':            [220, 220, 200],
    'picture frame':    [150, 100, 200],
    'mirror':           [180, 220, 220],
    'backpack':         [120, 80,  40 ],
    'suitcase':         [80,  60,  40 ],
    'pillow':           [220, 180, 200],
    'blanket':          [180, 160, 180],
    'shower':           [150, 200, 220],
    'bathtub':          [150, 180, 220],}
DEFAULT_COLOR = [80, 80, 80]
C0 = 0.28209479177387814


def load_ply(ply_path):
    ply = PlyData.read(str(ply_path))
    v = ply['vertex']
    xyz = np.stack([np.array(v['x']), np.array(v['y']), np.array(v['z'])], axis=1).astype(np.float64)
    return ply, xyz


def save_labeled_splat(ply_data, colors_01, labeled_mask, output_path):
    v = ply_data['vertex']
    v_arr = v.data.copy()

    f_dc = (colors_01 - 0.5) / C0
    v_arr['f_dc_0'][labeled_mask] = f_dc[labeled_mask, 0].astype(np.float32)
    v_arr['f_dc_1'][labeled_mask] = f_dc[labeled_mask, 1].astype(np.float32)
    v_arr['f_dc_2'][labeled_mask] = f_dc[labeled_mask, 2].astype(np.float32)

    for key in v_arr.dtype.names:
        if key.startswith('f_rest_'):
            v_arr[key][labeled_mask] = 0.0

    PlyData([PlyElement.describe(v_arr, 'vertex')], text=ply_data.text).write(str(output_path))


def backproject(u, v, depth, c2w, fx, fy, cx, cy):
    x_cam = (u - cx) * depth / fx
    y_cam = -(v - cy) * depth / fy
    z_cam = -depth
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(depth)], axis=1)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renders-dir',   required=True)
    parser.add_argument('--splat-ply',     required=True)
    parser.add_argument('--output-dir',    default='output')
    parser.add_argument('--yolo-model',    default='yolov8x-worldv2.pt')
    parser.add_argument('--sam-model',     default='sam2_b.pt')
    parser.add_argument('--conf',          type=float, default=0.15)
    parser.add_argument('--frame-skip',    type=int,   default=2)
    parser.add_argument('--pixel-stride',  type=int,   default=4)
    parser.add_argument('--knn-radius',    type=float, default=0.02)
    parser.add_argument('--min-votes',     type=int,   default=10)
    args = parser.parse_args()

    renders_dir = Path(args.renders_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(renders_dir / 'camera_params.json') as f:
        camera_params = json.load(f)

    print(f"Loading Gaussians from {args.splat_ply}...")
    ply_data, gauss_xyz = load_ply(args.splat_ply)
    N = len(gauss_xyz)
    print(f"  {N:,} Gaussians loaded")

    print("Building KD-tree...")
    tree = cKDTree(gauss_xyz)

    print(f"Loading YOLO-World: {args.yolo_model}")
    yolo_world = YOLO(args.yolo_model)
    yolo_world.set_classes(CLASSES)
    print(f"  {len(CLASSES)} classes set")

    sam_model = SAM(args.sam_model)
    print(f"Loaded SAM: {args.sam_model}")


    rgb_files = sorted((renders_dir / 'rgb').glob('*.png'))
    rgb_files = rgb_files[::args.frame_skip]
    print(f"Processing {len(rgb_files)} frames...")

    votes = defaultdict(Counter)

    for i, rgb_path in enumerate(rgb_files):
        stem = rgb_path.stem

        depth_path = renders_dir / 'depth' / f'{stem}.npy'
        pose_path  = renders_dir / 'poses' / f'{stem}.npy'

        if not depth_path.exists() or not pose_path.exists():
            continue
        if stem not in camera_params:
            continue

        img   = cv2.imread(str(rgb_path))
        depth = np.load(depth_path)
        c2w   = np.load(pose_path)

        params = camera_params[stem]
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']

        if img.shape[:2] != depth.shape:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        H, W = img.shape[:2]
        print(f"  [{i+1:4d}/{len(rgb_files)}] {stem}", end='\r')

        result = yolo_world(img, verbose=False, conf=args.conf)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        cls_ids    = result.boxes.cls.cpu().numpy().astype(int)
        confs      = result.boxes.conf.cpu().numpy()

        keep = confs >= args.conf
        boxes_xyxy = boxes_xyxy[keep]
        cls_ids    = cls_ids[keep]
        if len(boxes_xyxy) == 0:
            continue

        sam_result = sam_model(img, bboxes=boxes_xyxy.tolist(), verbose=False)[0]
        if sam_result.masks is None:
            continue
        masks = sam_result.masks.data.cpu().numpy()

        label_map = np.full((H, W), '', dtype=object)
        for mask_np, cls_id in zip(masks, cls_ids):
            label = CLASSES[int(cls_id)]
            mask_rs = cv2.resize(mask_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            label_map[mask_rs] = label

        ys, xs = np.where(label_map != '')
        if len(ys) == 0:
            continue

        sub = np.arange(0, len(ys), args.pixel_stride)
        ys, xs = ys[sub], xs[sub]

        pix_labels = label_map[ys, xs]
        pix_depth  = depth[ys, xs]

        valid = (pix_depth > 0.05) & np.isfinite(pix_depth)
        if not valid.any():
            continue

        ys, xs     = ys[valid], xs[valid]
        pix_labels = pix_labels[valid]
        pix_depth  = pix_depth[valid]

        pts_world = backproject(xs.astype(np.float64), ys.astype(np.float64),pix_depth.astype(np.float64), c2w, fx, fy, cx, cy,)

        idx_lists = tree.query_ball_point(pts_world, r=args.knn_radius)
        for label, idx_list in zip(pix_labels, idx_lists):
            for g_idx in idx_list:
                votes[g_idx][label] += 1

    print("\nAssigning labels to Gaussians...")
    colors_uint8 = np.tile(DEFAULT_COLOR, (N, 1)).astype(np.uint8)
    label_arr    = np.full(N, 'unlabeled', dtype=object)

    for g_idx, counter in votes.items():
        if not counter:
            continue
        best, best_count = counter.most_common(1)[0]
        total = sum(counter.values())
        if best_count >= args.min_votes and best_count / total > 0.5:
            label_arr[g_idx]    = best
            colors_uint8[g_idx] = LABEL_COLORS.get(best, DEFAULT_COLOR)

    print("\nLabel summary:")
    for lbl, count in Counter(label_arr).most_common():
        print(f"  {lbl:20s} {count:8,}  ({100*count/N:.1f}%)")

    out_splat = output_dir / 'labeled_splat.ply'
    colors_01 = colors_uint8.astype(np.float32) / 255.0
    labeled_mask = label_arr != 'unlabeled'
    save_labeled_splat(ply_data, colors_01, labeled_mask, out_splat)
    print(f"\nLabeled splat → {out_splat}")
    print("Drag labeled_splat.ply into SuperSplat to inspect.")


if __name__ == '__main__':
    main()