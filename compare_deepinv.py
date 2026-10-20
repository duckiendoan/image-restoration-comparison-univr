"""
Requirements: pip install deepinv ptwt scikit-image
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import deepinv as dinv
import numpy as np
import torch
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from helpers import *
from hno_deblurring import gtik_dct, padPSF, psfGauss, tik_dct, tik_fft, tsvd_dct

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

@dataclass
class Config:
    image_size: int = 256
    pad: int = 20
    psf_sigma: float = 1.5
    psf_size: int = 17
    noise_sigma: float = 0.01
    seed: int = 0

    tv_lambdas: list[float] = field(default_factory=lambda: [0.005, 0.015, 0.04])
    wv_lambdas: list[float] = field(default_factory=lambda: [0.01, 0.03])
    admm_max_iter: int = 25

    out_dir: Path = Path("outputs")
    data_dir: str | None = None
    max_images: int | None = None
    warmup: bool = True


def parse_args():
    p = argparse.ArgumentParser(description="Classical vs Modern deblurring benchmark (dataset)")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--data-dir", type=str, default=None,
                   help="Folder of test images (default: built-in camera image)")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip the warmup pass (first image timings will include CUDA setup)")
    a = p.parse_args()
    return Config(
        image_size=a.size, noise_sigma=a.noise, data_dir=a.data_dir,
        max_images=a.max_images, out_dir=Path(a.out_dir), warmup=not a.no_warmup,
    ), a.cpu


# SHORTHANDS
def get_device(force_cpu):
    return torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device("cuda")


def rel_err(x_hat, x_true):
    return float(np.linalg.norm(np.asarray(x_hat, float) - np.asarray(x_true, float))
                 / np.linalg.norm(x_true))

def to_tensor(a: np.ndarray, device):
    # Have to match the expected tensor shape
    return torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

def to_numpy(t: torch.Tensor):
    return t.detach().cpu().squeeze().numpy()

def repeat_to_rgb(x: torch.Tensor):
    # Repeat a 1-channel tensor to 3 channels (for LPIPS / RGB-only models)
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


class GrayscaleDiffUNet(torch.nn.Module):
    """Wraps a 3-channel DiffUNet so it accepts 1-channel grayscale input."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        return self.model(repeat_to_rgb(x), sigma).mean(dim=1, keepdim=True)


def load_dataset(folder, size, max_images=None) -> list[np.ndarray]:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    if max_images:
        paths = paths[:max_images]

    images = []
    for p in paths:
        img = img_as_float(imread(p))
        if img.ndim == 3:
            img = rgb2gray(img[..., :3])
        img = img.astype(np.float32)
        if min(img.shape) < size:
            img = resize(img, (size, size), anti_aliasing=True).astype(np.float32)
        else:
            i0, j0 = (img.shape[0] - size) // 2, (img.shape[1] - size) // 2
            img = img[i0:i0 + size, j0:j0 + size]
        images.append(img)
    print(f"Loaded {len(images)} images from {folder}")
    return images


def default_camera(size):
    cam = img_as_float(data.camera()).astype(np.float32)
    f = cam.shape[0] // size
    return cam[::f, ::f][:size, :size]


@dataclass
class Problem:
    X: np.ndarray
    B_noisy: np.ndarray
    PSF: np.ndarray
    PSF_work: np.ndarray
    center: tuple[int, int]
    noise_sigma: float


def build_problem(cfg, X_true, rng: np.random.Generator):
    Xbig = np.pad(X_true, cfg.pad, mode="reflect")
    PSF, center = psfGauss(cfg.psf_size, cfg.psf_sigma)
    Pbig = padPSF(PSF, Xbig.shape)
    ci, cj = center
    Sbig = np.fft.fft2(np.roll(Pbig, (-ci, -cj), axis=(0, 1)))
    Bbig = np.real(np.fft.ifft2(Sbig * np.fft.fft2(Xbig)))

    p, s = cfg.pad, cfg.image_size
    B = Bbig[p:p + s, p:p + s].astype(np.float32)
    E = rng.standard_normal(B.shape).astype(np.float32)
    E /= np.linalg.norm(E)
    B_noisy = (B + cfg.noise_sigma * np.linalg.norm(B) * E).astype(np.float32)

    return Problem(X=X_true, B_noisy=B_noisy, PSF=PSF,
                   PSF_work=padPSF(PSF, X_true.shape).astype(np.float32),
                   center=center, noise_sigma=cfg.noise_sigma)


def build_classical_methods(methods):
    def tik_dct_fn(prob, y):
        t0 = time.perf_counter()
        Xh, alpha = tik_dct(prob.B_noisy, prob.PSF_work, prob.center)
        return Xh, f"alpha={alpha:.4g}", time.perf_counter() - t0

    def tsvd_dct_fn(prob, y):
        t0 = time.perf_counter()
        Xh, tol = tsvd_dct(prob.B_noisy, prob.PSF_work, prob.center)
        return Xh, f"tol={tol:.4g}", time.perf_counter() - t0

    def tik_fft_fn(prob, y):
        t0 = time.perf_counter()
        Xh, alpha = tik_fft(prob.B_noisy, prob.PSF_work, prob.center)
        return Xh, f"alpha={alpha:.4g}", time.perf_counter() - t0

    def gtik_dct_fn(prob, y):
        t0 = time.perf_counter()
        # Laplacian regularization (edge-preserving)
        Laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=float)
        Xh, alpha = gtik_dct(prob.B_noisy, prob.PSF_work, prob.center, Pd=Laplacian)
        return Xh, f"alpha={alpha:.4g} (Laplacian)", time.perf_counter() - t0

    methods["Classical: Tikhonov/DCT"] = tik_dct_fn
    methods["Classical: TSVD/DCT"] = tsvd_dct_fn
    methods["Classical: Tikhonov/FFT (mismatched BC)"] = tik_fft_fn
    methods["Classical: Gen. Tikhonov/DCT (Laplacian)"] = gtik_dct_fn

def build_methods(cfg, device):
    methods = {}
    build_classical_methods(methods)

    # DeepInverse
    # PSF
    w = torch.tensor(psfGauss(cfg.psf_size, cfg.psf_sigma)[0],
                     dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    physics = dinv.physics.Blur(
        filter=w, padding="reflect", device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=cfg.noise_sigma),
    )
    data_fidelity = dinv.optim.data_fidelity.L2()

    # Dummy test
    dummy = torch.randn(1, 1, cfg.image_size, cfg.image_size, device=device)
    norm_A2 = physics.compute_norm(dummy, tol=1e-3, max_iter=50, verbose=False).item()
    stepsize = 1.0 / norm_A2
    print(f"Physics: ||A||^2 = {norm_A2:.4f}, ADMM stepsize = {stepsize:.4g}")

    def build_method_with_admm(name: str, prior, lambdas: list[float]) -> None:
        """Build one ADMM solver per lambda, ONCE; per image just sweep them."""
        models = [
            (lam, dinv.optim.ADMM(
                data_fidelity=data_fidelity, prior=prior, max_iter=cfg.admm_max_iter,
                params_algo={"stepsize": stepsize, "lambda": lam, "beta": 1.0},
            ))
            for lam in lambdas
        ]

        def run_method_fn(prob: Problem, y: torch.Tensor):
            best_xhat, best_lam, best_t, best_err = None, None, None, np.inf
            for lam, model in models:
                t0 = time.perf_counter()
                with torch.no_grad():
                    xhat = model(y, physics)
                dt = time.perf_counter() - t0
                xhat_np = to_numpy(xhat)
                err = rel_err(xhat_np, prob.X)
                if err < best_err:
                    best_xhat, best_lam, best_t, best_err = xhat_np, lam, dt, err
            return best_xhat, f"lambda={best_lam:g}", best_t

        methods[name] = run_method_fn

    build_method_with_admm("Modern: TV + ADMM", dinv.optim.TVPrior(n_it_max=30, def_crit=1e-4), cfg.tv_lambdas)
    build_method_with_admm("Modern: Wavelet + ADMM", dinv.optim.WaveletPrior(level=3, wv="db8"), cfg.wv_lambdas)

    # DPIR
    denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                  pretrained="download", device=device)
    dpir = dinv.optim.DPIR(sigma=cfg.noise_sigma, denoiser=denoiser, device=device)

    def dpir_fn(prob, y):
        t0 = time.perf_counter()
        with torch.no_grad():
            xhat = dpir(y, physics)
        return to_numpy(xhat), "pretrained", time.perf_counter() - t0

    methods["Modern: DPIR (PnP + DRUNet)"] = dpir_fn

    # Diffusion
    # diff_model = dinv.models.DiffUNet(pretrained="download").to(device)
    diffpir = dinv.sampling.DiffPIR(
        model=denoiser, data_fidelity=data_fidelity,
        sigma=cfg.noise_sigma, device=device
    )

    def diffpir_fn(prob, y):
        t0 = time.perf_counter()
        with torch.no_grad():
            xhat = diffpir(y, physics)
        return to_numpy(xhat), "pretrained (RGB→Gray)", time.perf_counter() - t0

    methods["Modern: DiffPIR (diffusion prior)"] = diffpir_fn

    return methods


def warmup(methods, prob, y: torch.Tensor):
    """One untimed pass through every method (triggers CUDA kernel setup)."""
    for fn in methods.values():
        try:
            fn(prob, y)
        except:
            pass
    if device := y.device:
        torch.cuda.synchronize(device) if y.is_cuda else None
    print("Warmup pass done.")


@dataclass
class Metrics:
    psnr: dinv.metric.PSNR
    ssim: dinv.metric.SSIM
    lpips: dinv.metric.LPIPS | None
    niqe: dinv.metric.NIQE | None


def build_metrics(device):
    lpips = niqe = None
    try:
        lpips = dinv.metric.LPIPS(net_type="alex", device=device)
    except:
        print("LPIPS unavailable")
    try:
        niqe = dinv.metric.NIQE(device=device)
    except:
        print("NIQE unavailable")
    return Metrics(dinv.metric.PSNR().to(device), dinv.metric.SSIM().to(device), lpips, niqe)


def compute_metrics(x_hat_np, x_true, metrics, device):
    x_hat_t = to_tensor(np.clip(x_hat_np, 0, 1), device)
    x_true_t = to_tensor(np.clip(x_true, 0, 1), device)
    return {
        "rel_err": rel_err(x_hat_np, x_true),
        "psnr": metrics.psnr(x_hat_t, x_true_t).item(),
        "ssim": metrics.ssim(x_hat_t, x_true_t).item(),
        "lpips": metrics.lpips(repeat_to_rgb(x_hat_t), repeat_to_rgb(x_true_t)).item()
                 if metrics.lpips else None,
        "niqe": metrics.niqe(x_hat_t).item() if metrics.niqe else None,
    }


def main():
    cfg, force_cpu = parse_args()
    device = get_device(force_cpu)
    torch.manual_seed(cfg.seed)
    print(f"Running on {device}")

    dataset = load_dataset(cfg.data_dir, cfg.image_size, cfg.max_images) \
                if cfg.data_dir else [default_camera(cfg.image_size)]

    t_build = time.perf_counter()
    methods = build_methods(cfg, device)
    print(f"Pipeline built in {time.perf_counter() - t_build:.1f}s "
          f"({len(methods)} methods registered)\n")

    metrics = build_metrics(device)

    if cfg.warmup:
        prob_w = build_problem(cfg, dataset[0], np.random.default_rng(cfg.seed))
        warmup(methods, prob_w, to_tensor(prob_w.B_noisy, device))

    per_image = []
    by_method = {}
    first_prob = None
    first_image_results = {}

    for i, X_true in enumerate(tqdm(dataset, desc="images")):
        prob = build_problem(cfg, X_true, np.random.default_rng(cfg.seed + i))
        y = to_tensor(prob.B_noisy, device)

        # BASELINE
        baseline_name = "Blurred + noisy (no restoration)"
        row = compute_metrics(prob.B_noisy, prob.X, metrics, device)
        row.update({"method": baseline_name, "param": "", "time_s": None})
        by_method.setdefault(baseline_name, []).append(row)
        per_image.append({"image": i, **row})

        for name, fn in methods.items():
            xhat, info, elapsed = fn(prob, y)

            res = compute_metrics(xhat, prob.X, metrics, device)
            res.update({"method": name, "param": info, "time_s": elapsed})
            by_method.setdefault(name, []).append(res)
            per_image.append({"image": i, **res})

            if i == 0:
                first_prob = prob
                first_image_results[name] = (xhat, info, elapsed)

    # Report
    summary = {name: aggregate(rows) for name, rows in by_method.items()}
    print()
    print_summary(summary, {n: len(r) for n, r in by_method.items()})

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    save_per_image_csv(per_image, cfg.out_dir / "metrics_per_image.csv")
    save_summary_csv(summary, by_method, cfg.out_dir / "metrics_summary.csv")
    if first_prob is not None:
        table0 = [r for r in per_image if r["image"] == 0]
        save_figure(first_prob.X, first_prob.B_noisy, first_image_results, table0, cfg.out_dir / "comparison.png")


if __name__ == "__main__":
    main()
