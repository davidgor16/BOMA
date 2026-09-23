"""Correct saved UTT-Mean PNG titles with an exact pixel-integrity audit.

Only the title rectangle above the plot frame is editable. Filenames, dimensions,
mode, DPI, and every pixel outside that rectangle are preserved. No chart is
regenerated and no numerical data are inferred from the image.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


def digest(data):
    return hashlib.sha256(data).hexdigest()


def correct_title(raw, title, font_path):
    original = Image.open(io.BytesIO(raw))
    original.load()
    if original.format != "PNG" or original.mode != "RGBA" or original.size != (2100, 1200):
        raise ValueError("Unrecognized figure layout; inspect it instead of guessing a title region.")
    before = np.asarray(original)
    dark = (before[:300, :, :3] < 128).all(axis=2)
    frame_rows = np.flatnonzero(dark.sum(axis=1) > original.width * 0.75)
    if not len(frame_rows):
        raise ValueError("Cannot locate the upper plot frame.")
    frame_top = int(frame_rows[0])
    if frame_top not in (108, 111):
        raise ValueError("Unexpected plot-frame position.")
    header = before[:frame_top - 3]
    ink = (header[:, :, :3] != 255).any(axis=2) & (header[:, :, 3] > 0)
    yy, xx = np.where(ink)
    if not len(yy) or not (header[:, :, 3] == 255).all():
        raise ValueError("Unexpected empty or transparent title area.")
    top, bottom = int(yy.min()) - 3, int(yy.max()) + 4
    if top < 0 or bottom >= frame_top - 3:
        raise ValueError("Title rectangle is not separated from the plot.")
    rectangle = (0, top, original.width, bottom)  # Half-open coordinates.
    font = ImageFont.truetype(str(font_path), size=50)
    box = font.getbbox(title)
    width, height = box[2] - box[0], box[3] - box[1]
    center_x = (int(xx.min()) + int(xx.max()) + 1) / 2
    x = round(center_x - width / 2) - box[0]
    y = round((top + bottom - height) / 2) - box[1]
    if x + box[0] < 0 or x + box[2] > original.width or y + box[1] < top or y + box[3] > bottom:
        raise ValueError("Replacement text does not fit the authorized rectangle.")
    edited = original.copy()
    drawing = ImageDraw.Draw(edited)
    drawing.rectangle((0, top, original.width - 1, bottom - 1), fill=(255, 255, 255, 255))
    drawing.text((x, y), title, font=font, fill=(0, 0, 0, 255))
    metadata = PngImagePlugin.PngInfo()
    for key, value in original.info.items():
        if isinstance(value, str):
            metadata.add_text(key, value)
    buffer = io.BytesIO()
    options = {"pnginfo": metadata}
    for key in ("dpi", "icc_profile", "exif"):
        if key in original.info:
            options[key] = original.info[key]
    edited.save(buffer, format="PNG", **options)
    corrected_raw = buffer.getvalue()
    reloaded = Image.open(io.BytesIO(corrected_raw))
    reloaded.load()
    if reloaded.size != original.size or reloaded.mode != original.mode or reloaded.info.get("dpi") != original.info.get("dpi"):
        raise ValueError("Dimensions, mode, or DPI changed.")
    changed = np.any(before != np.asarray(reloaded), axis=2)
    changed_inside = int(changed[top:bottom].sum())
    changed_outside = int(changed[:top].sum() + changed[bottom:].sum())
    if changed_outside != 0 or changed_inside == 0:
        raise ValueError("Pixel-integrity check failed.")
    return corrected_raw, {
        "original_sha256": digest(raw), "corrected_sha256": digest(corrected_raw),
        "original_bytes": len(raw), "corrected_bytes": len(corrected_raw),
        "title": title, "editable_rectangle_xyxy": list(rectangle),
        "changed_pixels_inside_title": changed_inside,
        "changed_pixels_outside_title": changed_outside,
        "dimensions": list(original.size), "mode": original.mode,
        "dpi": list(original.info.get("dpi", ())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.input_root.resolve() == args.output_root.resolve():
        parser.error("Use a separate output tree so the originals remain available.")
    if args.font is None:
        import matplotlib
        args.font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    paths = sorted(args.input_root.rglob("*_utt_score_dist.png"))
    if not paths:
        parser.error("No matching distribution figures found.")
    records, cache = [], {}
    for path in paths:
        relative = path.relative_to(args.input_root).as_posix()
        prefix = path.stem.removesuffix("_utt_score_dist").title()
        title = f"{prefix} - UTT-Mean Distribution"
        raw = path.read_bytes()
        key = (digest(raw), title)
        if key not in cache:
            cache[key] = correct_title(raw, title, args.font)
        corrected, record = cache[key]
        target = args.output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != corrected:
            raise FileExistsError(f"Different output already exists: {target}")
        target.write_bytes(corrected)
        if digest(target.read_bytes()) != record["corrected_sha256"]:
            raise ValueError(f"Saved output checksum mismatch: {target}")
        records.append({
            "source_path": relative,
            "path": relative.replace("CV-ML", "CV-DB").replace("CV_ML", "CV_DB"),
            **record,
        })
    report = {
        "operation": "Saved histogram title correction only: Utterance Total Score Distribution -> UTT-Mean Distribution",
        "images": len(records), "distinct_original_images": len(cache),
        "total_changed_pixels_outside_title": sum(r["changed_pixels_outside_title"] for r in records),
        "font_sha256": digest(args.font.read_bytes()),
        "verification": "RGBA pixels compared exactly after lossless PNG save and reload; all pixels outside the half-open title rectangle are identical",
        "files": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("images", "distinct_original_images", "total_changed_pixels_outside_title")}))


if __name__ == "__main__":
    main()
