"""
segmenter.py
============
Turns a drawing of ANY length of digits (e.g. "7", "2048", "9000000001")
into a batch of MNIST-style 28x28 images - one per digit - so a single-digit
CNN can read the whole number.

Pipeline
--------
1. Binarize the canvas                      (ink vs background)
2. Find connected strokes                   (scipy.ndimage.label)
3. Drop specks of noise
4. Merge strokes that sit on top of each other (e.g. the bar of a "5",
   an open "4") so one digit = one group
5. Split groups that are suspiciously wide  (two digits that touch)
6. Sort left -> right
7. Normalize every digit the way MNIST was made:
      - stroke thickness normalized (thin pens & fat brushes both work)
      - scaled to fit 20x20, keeping aspect ratio
      - centered by center-of-mass inside 28x28

This module has NO TensorFlow dependency, so it can be tested on its own.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

# ── Tunables ──────────────────────────────────────────────────────────────────
INK_THRESHOLD     = 60     # grayscale value (0-255) above which a pixel is "ink"
MIN_AREA_PX       = 25     # components smaller than this are always noise
NOISE_REL_SIZE    = 0.12   # ...or smaller than 12% of the tallest digit in both w & h
X_OVERLAP_MERGE   = 0.40   # merge strokes whose x-ranges overlap >= 40% of the narrower one
WIDE_RATIO_SPLIT  = 1.25   # a group wider than 1.25 x its height may be 2+ touching digits
SPLIT_SEARCH_LO   = 0.25   # only look for a cut in the middle 25%..75% of the group
SPLIT_SEARCH_HI   = 0.75
TARGET_STROKE_REL = 0.12   # MNIST stroke width is ~12% of the digit size (2.4px of 20px)
MAX_CANVAS_WIDTH  = 6000   # safety: shrink absurdly large uploads


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════
class _Group:
    """A set of connected components that together form one digit."""

    __slots__ = ("x0", "x1", "y0", "y1", "labels")

    def __init__(self, x0, x1, y0, y1, labels):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.labels = set(labels)

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def merge(self, other: "_Group"):
        self.x0 = min(self.x0, other.x0)
        self.x1 = max(self.x1, other.x1)
        self.y0 = min(self.y0, other.y0)
        self.y1 = max(self.y1, other.y1)
        self.labels |= other.labels


def _x_overlap(a: _Group, b: _Group) -> float:
    """Horizontal overlap as a fraction of the narrower group (0..1)."""
    ov = min(a.x1, b.x1) - max(a.x0, b.x0)
    if ov <= 0:
        return 0.0
    return ov / max(1, min(a.w, b.w))


def _tight_crop(mask: np.ndarray):
    """Crop a boolean mask to its bounding box. Returns (crop, x0, y0) or None."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return mask[y0:y1, x0:x1], int(x0), int(y0)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1-4: find the digit groups
# ══════════════════════════════════════════════════════════════════════════════
def _find_groups(mask: np.ndarray):
    labeled, n = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return labeled, []

    slices = ndi.find_objects(labeled)
    areas = ndi.sum(mask, labeled, index=np.arange(1, n + 1))

    comps = []
    for i, (sl, area) in enumerate(zip(slices, areas), start=1):
        ys, xs = sl
        comps.append(_Group(xs.start, xs.stop, ys.start, ys.stop, {i}))

    # ── Remove noise ──────────────────────────────────────────────────────
    tallest = max(c.h for c in comps)
    keep = []
    for c, area in zip(comps, areas):
        tiny_abs = area < MIN_AREA_PX
        tiny_rel = c.h < NOISE_REL_SIZE * tallest and c.w < NOISE_REL_SIZE * tallest
        if not (tiny_abs or tiny_rel):
            keep.append(c)
    if not keep:                      # everything looked like noise -> keep the biggest
        keep = [max(comps, key=lambda c: c.w * c.h)]

    # ── Merge strokes that overlap horizontally (one digit, several strokes) ──
    merged = True
    while merged:
        merged = False
        keep.sort(key=lambda g: g.x0)
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                if _x_overlap(keep[i], keep[j]) >= X_OVERLAP_MERGE:
                    keep[i].merge(keep[j])
                    del keep[j]
                    merged = True
                    break
            if merged:
                break

    keep.sort(key=lambda g: g.x0)
    return labeled, keep


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: split touching digits
# ══════════════════════════════════════════════════════════════════════════════
def _split_wide(mask_crop: np.ndarray, x_offset: int, depth: int = 0):
    """
    Recursively cut a wide blob at the thinnest vertical "neck".
    Returns a list of (x_start, x_end) column ranges relative to the full canvas.
    """
    h, w = mask_crop.shape
    if depth >= 8 or w <= WIDE_RATIO_SPLIT * h:
        return [(x_offset, x_offset + w)]

    profile = mask_crop.sum(axis=0).astype(float)
    profile = ndi.uniform_filter1d(profile, size=3, mode="nearest")

    lo = int(w * SPLIT_SEARCH_LO)
    hi = max(lo + 1, int(w * SPLIT_SEARCH_HI))
    window = profile[lo:hi]
    # Prefer the thinnest neck; break ties toward the centre.
    centre_penalty = np.abs(np.arange(lo, hi) - w / 2) / w * 0.5
    cut = lo + int(np.argmin(window + centre_penalty))

    left, right = mask_crop[:, :cut], mask_crop[:, cut:]
    if left.sum() == 0 or right.sum() == 0:
        return [(x_offset, x_offset + w)]

    return _split_wide(left, x_offset, depth + 1) + _split_wide(right, x_offset + cut, depth + 1)


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: MNIST-style normalization of ONE digit
# ══════════════════════════════════════════════════════════════════════════════
def _estimate_stroke_width(mask: np.ndarray) -> float:
    """Average pen width in pixels: w ~= 2 * area / perimeter."""
    area = float(mask.sum())
    edge = mask & ~ndi.binary_erosion(mask)
    perim = float(edge.sum())
    return 2.0 * area / perim if perim else 1.0


def _normalize_stroke(mask: np.ndarray) -> np.ndarray:
    """Make the pen thickness ~12% of the digit size, whatever brush was used."""
    crop = _tight_crop(mask)
    if crop is None:
        return mask
    m = crop[0]
    side = max(m.shape)
    target = max(2.0, TARGET_STROKE_REL * side)
    width = _estimate_stroke_width(m)

    pad = int(max(width, target)) + 4
    padded = np.pad(m, pad)

    if width < target * 0.9:                        # too thin -> thicken
        r = (target - width) / 2.0
        out = ndi.distance_transform_edt(~padded) <= r
    elif width > target * 1.1:                      # too fat  -> thin out
        r = (width - target) / 2.0
        out = ndi.distance_transform_edt(padded) > r
        if out.sum() < 0.2 * padded.sum():          # never erase the digit
            out = padded
    else:
        out = padded
    return out


def normalize_digit(mask: np.ndarray) -> np.ndarray:
    """
    boolean mask of ONE digit  ->  float32 array (28, 28), values 0..1,
    prepared exactly like the MNIST dataset (20x20 box, centred by mass).
    """
    mask = _normalize_stroke(mask)
    crop = _tight_crop(mask)
    if crop is None:
        return np.zeros((28, 28), dtype="float32")
    m = crop[0]
    h, w = m.shape

    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = Image.fromarray((m * 255).astype("uint8")).resize((nw, nh), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32")

    canvas = np.zeros((28, 28), dtype="float32")
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = arr

    # Centre by centre of mass (what MNIST did)
    if canvas.sum() > 0:
        cy, cx = ndi.center_of_mass(canvas)
        canvas = ndi.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant", cval=0.0)

    canvas = np.clip(canvas, 0, None)
    peak = canvas.max()
    if peak > 0:
        canvas /= peak
    return canvas.astype("float32")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def segment_digits(gray: np.ndarray):
    """
    gray : 2-D uint8 array (0-255), white-ish ink on a dark background.

    Returns a list (left -> right) of dicts:
        {"image": float32 (28,28), "box": [x0, y0, x1, y1]}
    An empty list means the canvas is blank.
    """
    mask = gray > INK_THRESHOLD
    if not mask.any():
        return []

    labeled, groups = _find_groups(mask)

    # Pixel mask of each group (only its own strokes, so neighbours never bleed in)
    pieces = []   # (group_mask_in_full_canvas_crop, x0, y0)
    for g in groups:
        sub_labels = labeled[g.y0:g.y1, g.x0:g.x1]
        gm = np.isin(sub_labels, list(g.labels))
        pieces.append((gm, g.x0, g.y0))

    # Split touching digits
    final = []
    for gm, x0, y0 in pieces:
        for xs, xe in _split_wide(gm, 0):
            part = gm[:, xs:xe]
            crop = _tight_crop(part)
            if crop is None:
                continue
            m, cx0, cy0 = crop
            final.append((m, x0 + xs + cx0, y0 + cy0))

    final.sort(key=lambda t: t[1])

    results = []
    for m, x0, y0 in final:
        h, w = m.shape
        results.append({
            "image": normalize_digit(m),
            "box": [int(x0), int(y0), int(x0 + w), int(y0 + h)],
        })
    return results


def load_gray(pil_image: Image.Image) -> np.ndarray:
    """PIL image (any mode, possibly with transparency) -> 2-D uint8 grayscale."""
    if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
        bg = Image.new("RGBA", pil_image.size, (0, 0, 0, 255))
        bg.alpha_composite(pil_image.convert("RGBA"))
        pil_image = bg
    pil_image = pil_image.convert("L")
    if pil_image.width > MAX_CANVAS_WIDTH:
        ratio = MAX_CANVAS_WIDTH / pil_image.width
        pil_image = pil_image.resize((MAX_CANVAS_WIDTH, max(1, int(pil_image.height * ratio))), Image.LANCZOS)
    return np.asarray(pil_image, dtype="uint8")
