"""Live ASCII mirror: your webcam, rendered as ASCII art in real time.

Install:  pip install opencv-python numpy
Run:      python ascii_webcam.py

Keys:  c = toggle color   i = invert   e = toggle contrast boost
       s = save snapshot   q / Esc = quit
"""
import argparse
import time

import cv2
import numpy as np

RAMP = " .:-=+*#%W"      # sparse -> dense (bright pixels get dense characters)
CELL_W, CELL_H = 8, 16   # pixel size of one character cell (2:1 like a terminal)


def build_atlas(chars):
    """Pre-draw each character once so frames can be assembled with pure numpy."""
    atlas = np.zeros((len(chars), CELL_H, CELL_W), dtype=np.uint8)
    for i, ch in enumerate(chars):
        if ch != " ":
            cv2.putText(atlas[i], ch, (0, CELL_H - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, 255, 1, cv2.LINE_AA)
    return atlas


def render_frame(frame, atlas, cols, color=True, invert=False, boost=True):
    """Turn a BGR camera frame into an ASCII-art BGR image."""
    h, w = frame.shape[:2]
    rows = max(1, int(cols * (h / w) * (CELL_W / CELL_H)))
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if boost:
        gray = cv2.equalizeHist(gray)

    n = len(atlas)
    idx = gray.astype(np.int32) * n // 256
    if invert:
        idx = n - 1 - idx

    # (rows, cols, H, W) -> (rows*H, cols*W)
    mask = atlas[idx].transpose(0, 2, 1, 3).reshape(rows * CELL_H, cols * CELL_W)

    if color:
        # brighten colors so thin glyph strokes stay vivid
        vivid = cv2.convertScaleAbs(small, alpha=1.4, beta=20)
        tint = cv2.resize(vivid, (cols * CELL_W, rows * CELL_H),
                          interpolation=cv2.INTER_NEAREST)
        out = (tint.astype(np.uint16) * mask[..., None] // 255).astype(np.uint8)
    else:
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return out


def main():
    p = argparse.ArgumentParser(description="Real-time webcam ASCII art")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--cols", type=int, default=110, help="characters across (default 110)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open the camera. Check permissions or try --camera 1")
        return

    atlas = build_atlas(RAMP)
    color, invert, boost = True, False, True
    last, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, like a real mirror
        out = render_frame(frame, atlas, args.cols, color, invert, boost)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(out, f"{fps:4.1f} fps", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("ASCII Mirror", out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            color = not color
        elif key == ord("i"):
            invert = not invert
        elif key == ord("e"):
            boost = not boost
        elif key == ord("s"):
            name = time.strftime("ascii_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, out)
            print("Saved", name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
