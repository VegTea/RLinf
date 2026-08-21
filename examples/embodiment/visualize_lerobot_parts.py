#!/usr/bin/env python3
"""Create lightweight visual checks for local LeRobot dataset parts."""

from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

import imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", help="LeRobot dataset part directories.")
    parser.add_argument("--output-dir", required=True, help="Directory for visualization outputs.")
    parser.add_argument("--episodes", default="0,50,100,last", help="Comma-separated episode ids per part.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--sheet-frames", type=int, default=8, help="Number of frames per contact sheet.")
    return parser.parse_args()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _episode_ids(spec: str, total_episodes: int) -> list[int]:
    ids = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "last":
            idx = total_episodes - 1
        else:
            idx = int(token)
        if idx < 0:
            idx += total_episodes
        if not (0 <= idx < total_episodes):
            raise ValueError(f"episode id {idx} out of range 0..{total_episodes - 1}")
        if idx not in ids:
            ids.append(idx)
    return ids


def _parquet_path(part: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunks_size
    return part / info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)


def _decode_image(value) -> Image.Image:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value.astype(np.uint8)).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _scalar(value):
    return value.as_py() if hasattr(value, "as_py") else value


def _overlay(image: Image.Image, lines: list[str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line_h = 14
    pad = 5
    width = max(draw.textlength(line) for line in lines) + pad * 2
    height = line_h * len(lines) + pad * 2
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255))
        y += line_h
    return canvas


def _concat_views(wrist: Image.Image, table: Image.Image, lines: list[str]) -> np.ndarray:
    wrist = wrist.resize((256, 256))
    table = table.resize((256, 256))
    canvas = Image.new("RGB", (512, 256), (0, 0, 0))
    canvas.paste(wrist, (0, 0))
    canvas.paste(table, (256, 0))
    return np.asarray(_overlay(canvas, lines))


def _episode_source(source_records: list[dict], episode_index: int) -> dict:
    if episode_index < len(source_records):
        return source_records[episode_index]
    return {}


def _make_video_and_sheet(part: Path, out_dir: Path, episode_index: int, args, source: dict) -> dict:
    info = _read_json(part / "meta" / "info.json")
    episodes = _read_jsonl(part / "meta" / "episodes.jsonl")
    episode = episodes[episode_index]
    table = pq.read_table(
        _parquet_path(part, info, episode_index),
        columns=[
            "observation.images.wrist",
            "observation.images.table",
            "goal.stage_id",
            "goal.step_index",
            "frame_index",
            "timestamp",
        ],
    )
    total = table.num_rows
    step = max(1, total // args.max_video_frames) if args.max_video_frames > 0 else 1
    frame_indices = list(range(0, total, step))
    if frame_indices[-1] != total - 1:
        frame_indices.append(total - 1)

    part_name = part.name
    stem = f"{part_name}_episode_{episode_index:06d}"
    video_path = out_dir / f"{stem}.mp4"
    sheet_path = out_dir / f"{stem}_sheet.jpg"
    source_name = Path(str(source.get("trajectory", ""))).name

    with imageio.get_writer(video_path, fps=int(args.fps), codec="libx264", quality=8) as writer:
        for idx in frame_indices:
            lines = [
                f"{part_name} ep {episode_index:03d} frame {idx:03d}/{total - 1:03d}",
                f"stage {_scalar(table['goal.stage_id'][idx])} goal {_scalar(table['goal.step_index'][idx])}",
                source_name[:64],
            ]
            writer.append_data(
                _concat_views(
                    _decode_image(table["observation.images.wrist"][idx]),
                    _decode_image(table["observation.images.table"][idx]),
                    lines,
                )
            )

    sheet_count = max(1, int(args.sheet_frames))
    sheet_indices = np.linspace(0, total - 1, sheet_count, dtype=int).tolist()
    sheet_frames = []
    for idx in sheet_indices:
        lines = [
            f"{part_name} ep {episode_index:03d}",
            f"frame {idx:03d}/{total - 1:03d}",
            f"stage {_scalar(table['goal.stage_id'][idx])}",
        ]
        sheet_frames.append(
            Image.fromarray(
                _concat_views(
                    _decode_image(table["observation.images.wrist"][idx]),
                    _decode_image(table["observation.images.table"][idx]),
                    lines,
                )
            )
        )
    sheet = Image.new("RGB", (512 * 2, 256 * ((len(sheet_frames) + 1) // 2)), (20, 20, 20))
    for i, frame in enumerate(sheet_frames):
        sheet.paste(frame, ((i % 2) * 512, (i // 2) * 256))
    sheet.save(sheet_path, quality=92)

    return {
        "part": part_name,
        "episode_index": int(episode_index),
        "length": int(episode["length"]),
        "frames_written": int(len(frame_indices)),
        "source_trajectory": source.get("trajectory"),
        "video": str(video_path),
        "sheet": str(sheet_path),
    }


def _write_html(out_dir: Path, records: list[dict]) -> None:
    rows = []
    for rec in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(rec['part'])}</td>"
            f"<td>{rec['episode_index']}</td>"
            f"<td>{rec['length']}</td>"
            f"<td><a href='{html.escape(Path(rec['video']).name)}'>mp4</a></td>"
            f"<td><a href='{html.escape(Path(rec['sheet']).name)}'>sheet</a></td>"
            f"<td>{html.escape(Path(str(rec.get('source_trajectory') or '')).name)}</td>"
            "</tr>"
        )
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>LeRobot Visualization</title>
<style>
body { font-family: sans-serif; margin: 24px; background: #f7f7f7; color: #222; }
table { border-collapse: collapse; width: 100%; background: white; }
td, th { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
th { background: #eee; text-align: left; }
a { color: #0645ad; }
</style></head><body>
<h1>LeRobot Visualization</h1>
<p>Each video concatenates wrist view on the left and table view on the right.</p>
<table><thead><tr><th>part</th><th>episode</th><th>length</th><th>video</th><th>sheet</th><th>source</th></tr></thead>
<tbody>
""" + "\n".join(rows) + """
</tbody></table></body></html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    args = parse_args()
    parts = [Path(p).expanduser().resolve() for p in args.parts]
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for part in parts:
        info = _read_json(part / "meta" / "info.json")
        source_records = _read_jsonl(part / "meta" / "source_replay.jsonl")
        for episode_index in _episode_ids(args.episodes, int(info["total_episodes"])):
            record = _make_video_and_sheet(
                part,
                out_dir,
                episode_index,
                args,
                _episode_source(source_records, episode_index),
            )
            print(json.dumps(record, ensure_ascii=True), flush=True)
            records.append(record)

    _write_json = out_dir / "summary.json"
    _write_json.write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_html(out_dir, records)
    print(json.dumps({"mode": "visualize_lerobot_parts", "output_dir": str(out_dir), "items": len(records)}), flush=True)


if __name__ == "__main__":
    main()
