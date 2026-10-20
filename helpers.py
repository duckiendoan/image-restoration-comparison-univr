import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_KEYS = ["rel_err", "psnr", "ssim", "lpips", "niqe", "time_s"]

def aggregate(rows: list[dict]):
    out = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
    return out


def print_summary(summary: dict, counts: dict):
    specs = {"rel_err": "8.4f", "psnr": "7.2f", "ssim": "6.3f",
             "lpips": "7.4f", "niqe": "7.3f", "time_s": "8.3f"}
    print(f"{'Method':40s} {'n':>3s}  " + "  ".join(f"{k:>17s}" for k in specs))
    print("-" * 150)
    for name, stats in summary.items():
        cells = []
        for k, spec in specs.items():
            m, s = stats[k]
            cells.append(f"{format(m, spec)}±{format(s, spec)}" if m is not None
                         else "--".rjust(17))
        print(f"{name:40s} {counts[name]:3d}  " + "  ".join(cells))


def save_summary_csv(summary, by_method, path: Path):
    fields = ["method", "n"] + [f"{k}_{ms}" for k in METRIC_KEYS for ms in ("mean", "std")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, stats in summary.items():
            row = {"method": name, "n": len(by_method[name])}
            for k, (m, s) in stats.items():
                row[f"{k}_mean"], row[f"{k}_std"] = m, s
            w.writerow(row)
    print(f"Saved: {path}")


def save_per_image_csv(per_image: list[dict], path: Path) -> None:
    fields = ["image", "method", "param"] + METRIC_KEYS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_image)
    print(f"Saved: {path}")


def save_figure(X: np.ndarray, B_noisy: np.ndarray, results: dict, table: list[dict], path: Path) -> None:
    n_show = 2 + len(results)
    ncols, nrows = 3, int(np.ceil(n_show / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(X, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[1].imshow(B_noisy, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Blurred + noisy\nPSNR={table[0]['psnr']:.2f} dB")

    for idx, row in enumerate(table[1:], start=2):
        if idx >= len(axes) or row["method"] not in results:
            break
        axes[idx].imshow(np.clip(results[row["method"]][0], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(f"{row['method']}\n({row['param']}, PSNR={row['psnr']:.2f})", fontsize=8)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Saved: {path}")
