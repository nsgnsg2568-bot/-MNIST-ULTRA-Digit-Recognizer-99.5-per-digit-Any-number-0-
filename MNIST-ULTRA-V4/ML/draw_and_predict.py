"""
draw_and_predict.py
===================
A simple drawing window using Tkinter where you can draw a digit
with your mouse and your trained model will predict it in real time.

Run with:
    python draw_and_predict.py
"""

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import os


# ── Config ────────────────────────────────────────────────────────────────────
CANVAS_SIZE   = 280          # pixels — the drawing area
BRUSH_RADIUS  = 12           # thickness of the pen
MODEL_PATH    = "saved_model/mnist_model.keras"
BG_COLOR      = "#111111"
FG_COLOR      = "#ffffff"
ACCENT        = "#00e5ff"


class DrawApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MNIST Digit Recognizer — Draw a digit!")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # ── Load model ────────────────────────────────────────────────────
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'.\n"
                "Run  main.py  first to train and save the model."
            )
        print("Loading model...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded ✓")

        # PIL image used as the off-screen drawing buffer
        self.pil_img  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=16, pady=10)

        # Title
        title_font = tkfont.Font(family="Courier New", size=14, weight="bold")
        tk.Label(
            self.root, text="✏  Draw a digit  (0 – 9)",
            font=title_font, bg="#1a1a2e", fg=ACCENT
        ).pack(**pad)

        # Drawing canvas
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg=BG_COLOR, cursor="crosshair",
            highlightthickness=2, highlightbackground=ACCENT
        )
        self.canvas.pack(padx=16)
        self.canvas.bind("<B1-Motion>",   self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_press)

        # Prediction display
        pred_frame = tk.Frame(self.root, bg="#1a1a2e")
        pred_frame.pack(fill="x", **pad)

        tk.Label(pred_frame, text="Prediction:", font=("Courier New", 11),
                 bg="#1a1a2e", fg="#aaaaaa").pack(side="left")

        self.pred_var = tk.StringVar(value="—")
        self.pred_label = tk.Label(
            pred_frame, textvariable=self.pred_var,
            font=tkfont.Font(family="Courier New", size=36, weight="bold"),
            bg="#1a1a2e", fg=ACCENT, width=3
        )
        self.pred_label.pack(side="left", padx=10)

        self.conf_var = tk.StringVar(value="")
        tk.Label(pred_frame, textvariable=self.conf_var,
                 font=("Courier New", 11), bg="#1a1a2e", fg="#aaaaaa").pack(side="left")

        # Confidence bars for all 10 digits
        bars_frame = tk.Frame(self.root, bg="#1a1a2e")
        bars_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.bar_vars  = []
        self.bar_lbls  = []
        BAR_W = 220

        for i in range(10):
            row = tk.Frame(bars_frame, bg="#1a1a2e")
            row.pack(fill="x", pady=1)

            tk.Label(row, text=str(i), font=("Courier New", 10),
                     bg="#1a1a2e", fg="#888888", width=2).pack(side="left")

            bar_bg = tk.Frame(row, bg="#333333", height=8, width=BAR_W)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg="#555555", height=8, width=0)
            bar_fill.place(x=0, y=0, height=8)
            self.bar_vars.append((bar_fill, BAR_W))

            pct_lbl = tk.Label(row, text="", font=("Courier New", 9),
                               bg="#1a1a2e", fg="#666666", width=6)
            pct_lbl.pack(side="left")
            self.bar_lbls.append(pct_lbl)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1a1a2e")
        btn_frame.pack(pady=(4, 16))

        btn_style = dict(
            font=("Courier New", 11, "bold"),
            bd=0, padx=20, pady=6, cursor="hand2"
        )
        tk.Button(btn_frame, text="  Clear  ", bg="#333344", fg="#ffffff",
                  activebackground="#555566", command=self._clear, **btn_style
                  ).pack(side="left", padx=8)

        tk.Button(btn_frame, text="  Predict  ", bg=ACCENT, fg="#000000",
                  activebackground="#00b8cc", command=self._predict, **btn_style
                  ).pack(side="left", padx=8)

    # ── Drawing ───────────────────────────────────────────────────────────
    def _on_press(self, event):
        self._draw_at(event.x, event.y)

    def _on_drag(self, event):
        self._draw_at(event.x, event.y)
        self._predict()          # live prediction while drawing

    def _draw_at(self, x, y):
        r = BRUSH_RADIUS
        # Draw on Tkinter canvas (visual)
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill=FG_COLOR, outline=FG_COLOR)
        # Draw on PIL buffer (used for model input)
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _clear(self):
        self.canvas.delete("all")
        self.pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.pred_var.set("—")
        self.conf_var.set("")
        for (bar_fill, _), lbl in zip(self.bar_vars, self.bar_lbls):
            bar_fill.place(x=0, y=0, width=0, height=8)
            lbl.config(text="", fg="#666666")

    # ── Prediction ────────────────────────────────────────────────────────
    def _predict(self):
        # Resize 280×280 → 28×28
        small = self.pil_img.resize((28, 28), Image.LANCZOS)
        arr   = np.array(small, dtype="float32") / 255.0
        arr   = arr.reshape(1, 784)

        probs   = self.model.predict(arr, verbose=0)[0]   # shape (10,)
        top     = int(np.argmax(probs))
        conf    = probs[top] * 100

        self.pred_var.set(str(top))
        self.conf_var.set(f"({conf:.1f}% confidence)")

        # Update bars
        for i, ((bar_fill, bar_w), lbl) in enumerate(zip(self.bar_vars, self.bar_lbls)):
            w = int(probs[i] * bar_w)
            color = ACCENT if i == top else "#444455"
            bar_fill.place(x=0, y=0, width=w, height=8)
            bar_fill.config(bg=color)
            lbl.config(
                text=f"{probs[i]*100:5.1f}%",
                fg=ACCENT if i == top else "#666666"
            )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = DrawApp(root)
    root.mainloop()