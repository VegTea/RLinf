#!/usr/bin/env python

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "rlinf/assets_isaaclab/all_setting/table_cam_circle_50.jsonl"


def normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    if norm <= 0.0:
        raise ValueError(f"Cannot normalize zero vector: {v}")
    return [x / norm for x in v]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rotation_matrix_to_quat_wxyz(matrix):
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat = normalize([w, x, y, z])
    # Keep a stable sign for easier diffs. q and -q encode the same rotation.
    if quat[0] < 0.0:
        quat = [-x for x in quat]
    return quat


def look_at_ros_quat(eye, target, world_up=(0.0, 0.0, 1.0)):
    # ROS optical camera convention in IsaacLab: +Z forward, -Y up.
    z_axis = normalize([target[i] - eye[i] for i in range(3)])
    x_axis = cross(z_axis, list(world_up))
    if dot(x_axis, x_axis) < 1e-10:
        x_axis = cross(z_axis, [0.0, 1.0, 0.0])
    x_axis = normalize(x_axis)
    y_axis = cross(z_axis, x_axis)

    # Columns are camera-frame basis vectors expressed in world frame.
    matrix = [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]
    return rotation_matrix_to_quat_wxyz(matrix)


def round_list(values, digits=8):
    return [round(float(value), digits) for value in values]


DEFAULT_TABLE_CAM_POS = [1.0, 0.0, 0.4]
DEFAULT_TABLE_CAM_ROT = [0.35355, -0.61237, -0.61237, 0.35355]
DEFAULT_TABLE_ASSET = "table_copper.usd"
DEFAULT_CUBE_FIELDS = {
    "cube_1_pos": [0.4, 0.0, 0.0203],
    "cube_1_rpy": [0.0, 0.0, 0.0],
    "cube_2_pos": [0.52, 0.08, 0.0203],
    "cube_2_rpy": [0.0, 0.0, 0.0],
    "cube_3_pos": [0.6, -0.1, 0.0203],
    "cube_3_rpy": [0.0, 0.0, 0.0],
}


def generate_default_record():
    return {
        "id": "cam_default_000",
        "table_asset": DEFAULT_TABLE_ASSET,
        **DEFAULT_CUBE_FIELDS,
        "table_cam_pos": DEFAULT_TABLE_CAM_POS,
        "table_cam_rot": DEFAULT_TABLE_CAM_ROT,
        "source": "isaaclab_default_table_cam",
    }


def generate_angle_offsets(count, span):
    if span is None:
        return [2.0 * math.pi * idx / count for idx in range(count)]
    if count <= 1:
        return [0.0]

    # Keep the exact center pose reserved for cam_default_000. The generated
    # poses cover the full interval while skipping zero to avoid a duplicate.
    half_span = span / 2.0
    left_count = count // 2
    right_count = count - left_count
    left_offsets = [
        -half_span + half_span * idx / left_count for idx in range(left_count)
    ]
    right_offsets = [
        half_span * (idx + 1) / right_count for idx in range(right_count)
    ]
    return left_offsets + right_offsets


def generate_records(count, center, radius, height, target, start_angle, clockwise, id_start=0, span=None):
    records = []
    direction = -1.0 if clockwise else 1.0
    for idx, angle_offset in enumerate(generate_angle_offsets(count, span)):
        theta = start_angle + direction * angle_offset
        eye = [
            center[0] + radius * math.cos(theta),
            center[1] + radius * math.sin(theta),
            height,
        ]
        quat = look_at_ros_quat(eye, target)
        records.append(
            {
                "id": f"cam_{idx + id_start:03d}",
                "table_asset": DEFAULT_TABLE_ASSET,
                **DEFAULT_CUBE_FIELDS,
                "table_cam_pos": round_list(eye),
                "table_cam_rot": round_list(quat),
                "theta_rad": round(float(theta), 10),
                "theta_deg": round(float(math.degrees(theta) % 360.0), 6),
                "look_at_target": round_list(target),
                "circle_center_xy": round_list(center),
                "radius": round(float(radius), 8),
                "height": round(float(height), 8),
                "span_deg": None if span is None else round(float(math.degrees(span)), 8),
            }
        )
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate table_cam poses on a circle around the IsaacLab stack table center."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--center-x", type=float, default=0.5)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--height", type=float, default=0.4)
    parser.add_argument(
        "--target-x",
        type=float,
        default=None,
        help="Look-at target x. Defaults to --center-x for backward compatibility.",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=None,
        help="Look-at target y. Defaults to --center-y for backward compatibility.",
    )
    parser.add_argument("--target-z", type=float, default=0.11132162)
    parser.add_argument("--start-angle-deg", type=float, default=0.0)
    parser.add_argument(
        "--span-deg",
        type=float,
        default=None,
        help="Generate poses only within this angular span centered at --start-angle-deg. Defaults to 360.",
    )
    parser.add_argument("--clockwise", action="store_true")
    parser.add_argument(
        "--include-default-first",
        action="store_true",
        help="Keep the original IsaacLab table_cam pose as the first record.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.radius <= 0.0:
        raise ValueError("--radius must be positive")
    if args.span_deg is not None and not (0.0 < args.span_deg <= 360.0):
        raise ValueError("--span-deg must be in (0, 360]")

    generated_count = args.count - 1 if args.include_default_first else args.count
    if generated_count <= 0:
        raise ValueError("--count must be greater than 1 when --include-default-first is used")

    target = [
        args.center_x if args.target_x is None else args.target_x,
        args.center_y if args.target_y is None else args.target_y,
        args.target_z,
    ]
    records = []
    if args.include_default_first:
        records.append(generate_default_record())
    records.extend(
        generate_records(
        count=generated_count,
        center=[args.center_x, args.center_y],
        radius=args.radius,
        height=args.height,
        target=target,
        start_angle=math.radians(args.start_angle_deg),
        clockwise=args.clockwise,
        id_start=1 if args.include_default_first else 0,
        span=None if args.span_deg is None else math.radians(args.span_deg),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(f"Wrote {len(records)} camera poses to {args.output}")


if __name__ == "__main__":
    main()
