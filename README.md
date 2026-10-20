# Image Deblurring: A Comparison between Classical and Modern Methods
*Project done as part of the minicourse Image restoration: from model to learning based techniques taught by Prof. Mauro Dalla Mura at University of Verona.*

A comparative study of classical spectral filtering methods (based on the HNO book) and modern deep learning-based approaches (using the `DeepInverse` library) for image deblurring.

## Methods compared
- **Classical**: Tikhonov, TSVD, Generalized Tikhonov with DCT/FFT.
- **Modern methods** via `DeepInverse`: TV + ADMM, Wavelet + ADMM, Plug-and-Play (PnP) methods with Deep Learning (DPIR, DiffPIR)
- **Metrics**: PSNR, SSIM, LPIPS, NIQE, Relative Error, and Inference Time evaluated on a set of 12 grayscale images from [DPIR](https://github.com/cszn/DPIR/tree/master/testsets/set12).

## Installation
```bash
pip install numpy scipy matplotlib imageio scikit-image deepinv ptwt tqdm
```

## Usage
Run the benchmarking script to compare methods on a dataset of images:
```bash
python compare_deepinv.py --size 256 --noise 0.03 --data-dir /path/to/images
```
*(Omit `--data-dir` to use the built-in `skimage.data.camera` image).*

Results, including per-image metrics and visual comparisons, will be saved to the `outputs/` directory.

## References
- P. C. Hansen, J. G. Nagy, and D. P. O'Leary, *Deblurring Images: Matrices, Spectra, and Filtering*. SIAM, 2006.
- J. Tachella et al., *DeepInverse: A Python package for solving imaging inverse problems with deep learning*. JOSS, 2025.
