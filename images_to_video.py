"""Combine result images into a constant-frame-rate MP4.

Usage: python images_to_video.py --fps 10 --output results_video.mp4
Optional: --folder results
Dependencies: python -m pip install av pillow

Timestamp names such as 37_1.png are sorted numerically (37.1 seconds).
Other filenames use natural ordering. Every image contributes one frame,
regardless of the time gap in its filename. Duplicate timestamps are retained.
Images must have matching dimensions; odd dimensions get one black padding
pixel on the right/bottom for H.264. Existing output files are not overwritten.
"""

import argparse
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import av
from PIL import Image


def image_sort_key(path):
    """Sort timestamp stems numerically, then other names in natural order."""
    if re.fullmatch(r"\d+(?:[_.]\d+)?", path.stem):
        return (0, Decimal(path.stem.replace("_", ".")), path.name.casefold())
    parts = tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )
    return (1, parts, path.name.casefold())


def images_to_video(folder, output, fps=10):
    """Encode every supported image in a folder, publishing only on success."""
    rate = Fraction(str(fps))
    if rate <= 0:
        raise ValueError("Frame rate must be positive.")
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image directory does not exist: {folder}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions),
        key=image_sort_key,
    )
    if not images:
        raise ValueError(f"No supported images found in {folder}.")
    output = Path(output).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have the .mp4 extension.")
    if not output.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output.parent}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    with Image.open(images[0]) as first:
        size = first.size
    encoded_size = tuple(dimension + dimension % 2 for dimension in size)
    with TemporaryDirectory(prefix="images-video-", dir=output.parent) as directory:
        temporary = Path(directory) / "video.mp4"
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.width, stream.height = encoded_size
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "veryfast"}
            for index, path in enumerate(images):
                with Image.open(path) as image:
                    if image.size != size:
                        raise ValueError(
                            f"Image {path.name} has size {image.size}; expected {size}."
                        )
                    rgb = Image.new("RGB", encoded_size, "black")
                    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                        rgba = image.convert("RGBA")
                        rgb.paste(rgba, (0, 0), rgba)
                    else:
                        rgb.paste(image.convert("RGB"), (0, 0))
                    frame = av.VideoFrame.from_image(rgb)
                frame.pts = index
                frame.time_base = 1 / rate
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temporary.rename(output)
    print(
        f"Saved {len(images)} images at {float(rate):g} fps "
        f"({float(len(images) / rate):.3f} s) to {output}"
    )
    return len(images)


def main():
    images_to_video(Path(__file__).resolve().parent / "results", 
                    "results_video.mp4", 
                    2.0)  # fps


if __name__ == "__main__":
    main()
