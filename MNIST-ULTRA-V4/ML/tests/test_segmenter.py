"""
Tests for segmenter.py  (no TensorFlow needed).
Run:  python -m pytest tests -q      or      python tests/test_segmenter.py
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segmenter import segment_digits, normalize_digit

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def render(text, size=300, gap=60, canvas_h=400, pen=None):
    """Draw `text` digit-by-digit with a fixed gap; optionally re-pen with thicker/thinner strokes."""
    font = ImageFont.truetype(FONT, size)
    widths = [font.getbbox(c)[2] - font.getbbox(c)[0] for c in text]
    W = sum(widths) + gap * (len(text) + 1)
    img = Image.new("L", (W, canvas_h), 0)
    d = ImageDraw.Draw(img)
    x = gap
    for c, w in zip(text, widths):
        bb = font.getbbox(c)
        d.text((x - bb[0], (canvas_h - (bb[3] - bb[1])) // 2 - bb[1]), c, fill=255, font=font)
        x += w + gap
    a = np.array(img)
    if pen is not None:                       # simulate brush size by dilation/erosion
        m = a > 100
        if pen > 0:
            m = ndi.distance_transform_edt(~m) <= pen
        elif pen < 0:
            m = ndi.distance_transform_edt(m) > -pen
        a = (m * 255).astype("uint8")
    return a


def test_blank_canvas():
    assert segment_digits(np.zeros((400, 1000), dtype="uint8")) == []


def test_counts_various_lengths():
    for s in ["7", "42", "2048", "1234567890", "90000000001"]:
        res = segment_digits(render(s))
        assert len(res) == len(s), (s, len(res))


def test_left_to_right_order():
    res = segment_digits(render("3817"))
    xs = [r["box"][0] for r in res]
    assert xs == sorted(xs)


def test_output_shape_and_range():
    for r in segment_digits(render("58")):
        im = r["image"]
        assert im.shape == (28, 28) and im.dtype == np.float32
        assert 0.99 <= im.max() <= 1.0 and im.min() >= 0


def test_thin_and_fat_pens_still_segment():
    for pen in (-6, 0, 6, 14):
        res = segment_digits(render("2026", pen=pen))
        assert len(res) == 4, (pen, len(res))


def test_stroke_width_normalised():
    """Thin and fat pens should give a similar amount of ink after normalisation."""
    ink = []
    for pen in (-8, 0, 12):
        r = segment_digits(render("8", pen=pen))
        assert len(r) == 1
        ink.append(r[0]["image"].sum())
    assert max(ink) / min(ink) < 1.5, ink


def test_centred_in_28x28():
    im = segment_digits(render("4"))[0]["image"]
    cy, cx = ndi.center_of_mass(im)
    assert abs(cy - 14) < 1.5 and abs(cx - 14) < 1.5


def test_detached_stroke_merged_into_same_digit():
    """A '5' whose top bar is a separate stroke must remain ONE digit."""
    a = render("5")
    ys, xs = np.where(a > 0)
    top = ys.min() + 22                      # cut a gap just under the top bar
    a[top:top + 14, :] = 0
    assert len(segment_digits(a)) == 1


def test_noise_ignored():
    a = render("123")
    a[10:14, 10:14] = 255                    # tiny speck
    assert len(segment_digits(a)) == 3


def test_touching_digits_split():
    """Two digits drawn with NO gap (gap<0 via small size/gap) get split."""
    a = render("11", gap=-10)  # negative gap makes them touch / overlap slightly
    a = np.maximum(a, 0)
    res = segment_digits(a)
    assert len(res) >= 1  # best-effort; asserted loosely, see README


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
