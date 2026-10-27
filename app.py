import json
import os

import numpy as np
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

with open(os.path.join(BASE, "data", "params.json")) as f:
    PARAMS = json.load(f)
CHARGES = np.array(PARAMS["charges"])
MU, SIGMA = PARAMS["mean"], PARAMS["std"]

DEFAULTS = {"metode": "bootstrap", "n_pasien": 100, "anggaran": 1_400_000, "n_sim": 10_000, "seed": 301240034}
LIMITS = {"n_pasien": (1, 1000), "anggaran": (1, 1_000_000_000), "n_sim": (1000, 50_000)}


@app.template_filter("usd")
def usd(x):
    return "US$ " + f"{x:,.0f}".replace(",", ".")


def simulate(metode, n_pasien, n_sim, seed):
    """Menghasilkan total klaim n_pasien untuk n_sim skenario (Monte Carlo)."""
    rs = np.random.RandomState(seed)
    rows = max(1, 2_000_000 // n_pasien)          # proses per potongan agar hemat memori
    out = []
    done = 0
    while done < n_sim:
        r = min(rows, n_sim - done)
        if metode == "normal":
            klaim = rs.normal(MU, SIGMA, size=(r, n_pasien))
        else:
            klaim = rs.choice(CHARGES, size=(r, n_pasien), replace=True)
        out.append(klaim.sum(axis=1))
        done += r
    return np.concatenate(out)


def parse(form):
    vals, err = {}, None
    vals["metode"] = form.get("metode", "bootstrap")
    if vals["metode"] not in ("normal", "bootstrap"):
        return vals, "Metode tidak dikenal."
    labels = {"n_pasien": "Jumlah pasien", "anggaran": "Anggaran", "n_sim": "Jumlah skenario", "seed": "Seed"}
    for k in ("n_pasien", "anggaran", "n_sim", "seed"):
        raw = form.get(k, "").strip().replace(".", "").replace(",", "")
        try:
            v = int(raw)
        except ValueError:
            return vals, f"{labels[k]} harus berupa bilangan bulat."
        if k in LIMITS and not (LIMITS[k][0] <= v <= LIMITS[k][1]):
            return vals, f"{labels[k]} harus antara {LIMITS[k][0]:,} dan {LIMITS[k][1]:,}.".replace(",", ".")
        if k == "seed" and not (0 <= v < 2**32):
            return vals, "Seed harus antara 0 dan 4294967295."
        vals[k] = v
    return vals, err


@app.route("/", methods=["GET", "POST"])
def index():
    form, hasil, error = dict(DEFAULTS), None, None
    if request.method == "POST":
        form, error = parse(request.form)
        if error is None:
            total = simulate(form["metode"], form["n_pasien"], form["n_sim"], form["seed"])
            counts, edges = np.histogram(total, bins=40)
            hasil = {
                "prob": float((total > form["anggaran"]).mean()),
                "ekspektasi": form["n_pasien"] * MU,
                "rata": float(total.mean()), "std": float(total.std()),
                "p95": float(np.percentile(total, 95)), "p99": float(np.percentile(total, 99)),
                "counts": counts.tolist(), "edges": [round(float(e), 2) for e in edges],
            }
    return render_template("index.html", form=form, hasil=hasil, error=error, params=PARAMS)


if __name__ == "__main__":
    app.run(debug=True)
