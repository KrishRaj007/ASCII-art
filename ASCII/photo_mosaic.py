"""Build a big picture out of many small pictures.

Usage:
    python photo_mosaic.py target.jpg tiles_folder -c 80 -t 24 -o mosaic.png

target.jpg    the picture you want to recreate
tiles_folder  a folder with lots of images (the more, the better: 200+)
"""
import argparse
import os

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_tiles(folder, tile_size):
    """Center-crop each image to a square, shrink it, and record its average color."""
    tiles, means = [], []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(EXTS):
            continue
        try:
            img = Image.open(os.path.join(folder, name)).convert("RGB")
        except OSError:
            continue
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize(
            (tile_size, tile_size), Image.LANCZOS
        )
        tiles.append(img)
        means.append(np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0))
    return tiles, np.array(means)


def build_mosaic(target_path, tiles, means, cols, tile_size, tint, variety):
    target = Image.open(target_path).convert("RGB")
    w, h = target.size
    rows = max(1, round(cols * h / w))

    # Shrinking with BOX gives each pixel the exact average color of its cell.
    cells = np.asarray(target.resize((cols, rows), Image.BOX), dtype=np.float32)

    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size))
    rng = np.random.default_rng()
    k = max(1, min(variety, len(tiles)))

    for r in range(rows):
        for c in range(cols):
            color = cells[r, c]
            dist = ((means - color) ** 2).sum(axis=1)
            # pick randomly among the k closest tiles so one image isn't spammed
            best = np.argpartition(dist, k - 1)[:k] if k < len(tiles) else np.arange(len(tiles))
            tile = tiles[int(rng.choice(best))]
            if tint > 0:  # nudge the tile toward the target color for a sharper result
                overlay = Image.new("RGB", tile.size, tuple(int(x) for x in color))
                tile = Image.blend(tile, overlay, tint)
            canvas.paste(tile, (c * tile_size, r * tile_size))
    return canvas


def main():
    p = argparse.ArgumentParser(description="Photo mosaic maker")
    p.add_argument("target", help="image to recreate")
    p.add_argument("tiles", help="folder of small images to use as tiles")
    p.add_argument("-c", "--cols", type=int, default=80, help="tiles across (default 80)")
    p.add_argument("-t", "--tile-size", type=int, default=24, help="pixels per tile (default 24)")
    p.add_argument("--tint", type=float, default=0.25,
                   help="0 = pure tiles, 1 = flat color; 0.2-0.4 looks best")
    p.add_argument("--variety", type=int, default=5,
                   help="choose randomly among the N closest tiles (more = less repetition)")
    p.add_argument("-o", "--output", default="mosaic.png")
    args = p.parse_args()

    tiles, means = load_tiles(args.tiles, args.tile_size)
    if not tiles:
        print("No images found in", args.tiles)
        return
    if len(tiles) < 50:
        print(f"Only {len(tiles)} tiles found. It works, but 200+ looks much better.")

    mosaic = build_mosaic(args.target, tiles, means, args.cols, args.tile_size,
                          args.tint, args.variety)
    mosaic.save(args.output)
    print(f"Saved {args.output} ({mosaic.size[0]}x{mosaic.size[1]}px, "
          f"{len(tiles)} unique tiles)")


if __name__ == "__main__":
    main()
