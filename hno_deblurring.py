"""
hno_deblurring.py
------------------
Python port of the MATLAB functions accompanying:

    Deblurring Images - Matrices, Spectra, and Filtering
    P. C. Hansen, J. G. Nagy, and D. P. O'Leary
    SIAM, Philadelphia, 2006.

WITH THE HELP OF CLAUDE CODE.

CONVENTION CHANGE FROM THE MATLAB CODE
=======================================
MATLAB is 1-indexed; Python/NumPy is 0-indexed. Throughout this file the
PSF ``center`` argument is therefore expressed as a 0-based (row, col) tuple,
e.g. ``center = (12, 12)`` instead of MATLAB's ``center = [13, 13]``.
All the index arithmetic below has been re-derived for 0-based indexing
(it is *not* simply "subtract 1 and hope").

FUNCTIONS THAT ALREADY EXIST IN PYTHON LIBRARIES (NOT reimplemented here)
==========================================================================
MATLAB function          Python equivalent
------------------------ --------------------------------------------------
fft2 / ifft2              numpy.fft.fft2 / numpy.fft.ifft2
                           (or scipy.fft.fft2 / scipy.fft.ifft2)
dct2 / idct2               scipy.fft.dctn(x, type=2, norm='ortho') /
                           scipy.fft.idctn(x, type=2, norm='ortho')
                           NOTE: MATLAB's dct2/idct2 (Image Processing
                           Toolbox) use the *orthonormal* DCT-II, which is
                           exactly scipy.fft's norm='ortho' option. So the
                           book's own fallback functions dcts/dcts2/idcts/
                           idcts2 (hand-rolled DCT via FFT, for users
                           without the toolbox) are not needed in Python
                           at all -- scipy.fft.dctn/idctn cover both roles.
svd                        numpy.linalg.svd  (or scipy.linalg.svd)
svds (few largest SVs)     scipy.sparse.linalg.svds, or just numpy.linalg.svd
                           and take the first few components (fine for the
                           small PSF arrays used here)
toeplitz                   scipy.linalg.toeplitz
hankel                     scipy.linalg.hankel
fminbnd                    scipy.optimize.minimize_scalar(method='bounded')
imread                     skimage.io.imread / imageio.imread / PIL.Image.open
rgb2gray                   skimage.color.rgb2gray
mat2gray                   skimage.exposure.rescale_intensity (or a 3-line
                           manual min-max rescale, included below as
                           `mat2gray` for convenience)
imwrite                    skimage.io.imsave

Functions below ARE reimplemented because they encode book-specific
algorithms with no off-the-shelf equivalent:
    padPSF, psfGauss, psfDefocus, dctshift, kronDecomp,
    tik_fft, tik_dct, tik_sep, tsvd_fft, tsvd_dct, tsvd_sep,
    gtik_fft, gcv_tik, gcv_tsvd, gcv_gtik
"""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.linalg import svd, toeplitz, hankel
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def mat2gray(a):
    """Rescale array to [0, 1], the way MATLAB's mat2gray does."""
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    if hi == lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def dct2(x):
    """2-D orthonormal DCT-II, equivalent to MATLAB's dct2."""
    return dctn(x, type=2, norm='ortho')


def idct2(x):
    """2-D orthonormal inverse DCT-II, equivalent to MATLAB's idct2."""
    return idctn(x, type=2, norm='ortho')


# ---------------------------------------------------------------------------
# padPSF.m
# ---------------------------------------------------------------------------

def padPSF(PSF, shape):
    """
    Pad PSF with zeros (bottom/right) to make it an m-by-n array.

    Parameters
    ----------
    PSF : 2-D ndarray
    shape : (m, n) tuple, or scalar m (=> n = m)

    Returns
    -------
    P : (m, n) ndarray, PSF placed in the top-left corner, zeros elsewhere.
    """
    if np.isscalar(shape):
        m = n = int(shape)
    else:
        m, n = shape
    P = np.zeros((m, n))
    P[:PSF.shape[0], :PSF.shape[1]] = PSF
    return P


# ---------------------------------------------------------------------------
# psfGauss.m
# ---------------------------------------------------------------------------

def psfGauss(dim, s=2.0):
    """
    Construct a Gaussian blur PSF.

    Parameters
    ----------
    dim : int or (m, n)
    s : float or (s_row, s_col), standard deviations. Default 2.0.

    Returns
    -------
    PSF : (m, n) ndarray, normalized to sum to 1.
    center : (row, col) 0-based index of the PSF's peak.
    """
    if np.isscalar(dim):
        m = n = int(dim)
    else:
        m, n = dim
    if np.isscalar(s):
        s = (s, s)

    x = np.arange(-(n // 2), n - n // 2)
    y = np.arange(-(m // 2), m - m // 2)
    X, Y = np.meshgrid(x, y)  # default 'xy' indexing matches MATLAB meshgrid

    PSF = np.exp(-(X ** 2) / (2 * s[1] ** 2) - (Y ** 2) / (2 * s[0] ** 2))
    PSF = PSF / PSF.sum()

    center = np.unravel_index(np.argmax(PSF), PSF.shape)
    return PSF, center


# ---------------------------------------------------------------------------
# psfDefocus.m
# ---------------------------------------------------------------------------

def psfDefocus(dim, R=None):
    """
    Construct an out-of-focus (defocus) blur PSF: uniform disk of radius R.

    Parameters
    ----------
    dim : int or (m, n)
    R : float, radius. Default: min(center) (largest disk that fits).

    Returns
    -------
    PSF : (m, n) ndarray, normalized to sum to 1.
    center : (row, col) 0-based center index.
    """
    if np.isscalar(dim):
        m = n = int(dim)
    else:
        m, n = dim

    # 0-based center, equivalent to MATLAB's fix((dim+1)/2) - 1
    ci = (m - 1) // 2
    cj = (n - 1) // 2
    center = (ci, cj)

    if R is None:
        R = min(center)

    if R == 0:
        PSF = np.zeros((m, n))
        PSF[ci, cj] = 1.0
        return PSF, center

    PSF = np.ones((m, n)) / (np.pi * R * R)
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    mask = (ii - ci) ** 2 + (jj - cj) ** 2 > R ** 2
    PSF[mask] = 0.0
    PSF = PSF / PSF.sum()
    return PSF, center


# ---------------------------------------------------------------------------
# dctshift.m
# ---------------------------------------------------------------------------

def dctshift(PSF, center):
    """
    Create the array whose 2-D DCT gives the eigenvalues of the
    reflective-boundary blurring matrix (Toeplitz-plus-Hankel case).

    Parameters
    ----------
    PSF : (m, n) ndarray
    center : (row, col) 0-based center of the PSF.

    Returns
    -------
    Ps : (m, n) ndarray
    """
    m, n = PSF.shape
    ci, cj = center

    k = min(ci, m - 1 - ci, cj, n - 1 - cj)

    PP = PSF[ci - k:ci + k + 1, cj - k:cj + k + 1]

    Z1 = np.diag(np.ones(k + 1), k)[: 2 * k + 1, : 2 * k + 1]
    Z2 = np.diag(np.ones(k), k + 1)[: 2 * k + 1, : 2 * k + 1]

    PP = Z1 @ PP @ Z1.T + Z1 @ PP @ Z2.T + Z2 @ PP @ Z1.T + Z2 @ PP @ Z2.T

    Ps = np.zeros((m, n))
    Ps[:2 * k + 1, :2 * k + 1] = PP
    return Ps


# ---------------------------------------------------------------------------
# kronDecomp.m  (+ buildToep / buildCirc / buildHank helpers)
# ---------------------------------------------------------------------------

def _buildToep(c, k):
    """Banded Toeplitz matrix from vector c, 0-based center index k."""
    n = len(c)
    col = np.zeros(n)
    row = np.zeros(n)
    col[:n - k] = c[k:n]
    row[:k + 1] = c[k::-1]
    return toeplitz(col, row)


def _buildCirc(c, k):
    """Banded circulant matrix from vector c, 0-based center index k."""
    n = len(c)
    col = np.concatenate([c[k:n], c[0:k]])
    row = np.concatenate([c[k::-1], c[n - 1:k:-1]])
    return toeplitz(col, row)


def _buildHank(c, k):
    """Hankel matrix (reflexive BC correction) from vector c, 0-based k."""
    n = len(c)
    col = np.zeros(n)
    col[:n - k - 1] = c[k + 1:n]
    row = np.zeros(n)
    row[n - k:n] = c[0:k]
    return hankel(col, row)


def kronDecomp(P, center, BC='zero'):
    """
    Kronecker product decomposition of a PSF array: A ~= kron(Ar, Ac).

    Parameters
    ----------
    P : (m, n) ndarray, the PSF.
    center : (row, col) 0-based center of the PSF.
    BC : 'zero', 'reflexive', or 'periodic'.

    Returns
    -------
    Ar, Ac : ndarrays such that kron(Ar, Ac) approximates (or, for
             separable PSFs, exactly equals) the blurring matrix A.
    """
    ci, cj = center

    U, S, Vt = svd(P)
    V = Vt.T
    if S[1] / S[0] > np.sqrt(np.finfo(float).eps):
        import warnings
        warnings.warn('The PSF, P is not separable; using separable approximation.')

    # Make the dominant singular vectors nonnegative (PSF has nonneg entries)
    minU = abs(U[:, 0].min())
    maxU = abs(U[:, 0]).max()
    if minU == maxU:
        U = -U
        V = -V

    c = np.sqrt(S[0]) * U[:, 0]
    r = np.sqrt(S[0]) * V[:, 0]

    if BC == 'zero':
        Ar = _buildToep(r, cj)
        Ac = _buildToep(c, ci)
    elif BC == 'reflexive':
        Ar = _buildToep(r, cj) + _buildHank(r, cj)
        Ac = _buildToep(c, ci) + _buildHank(c, ci)
    elif BC == 'periodic':
        Ar = _buildCirc(r, cj)
        Ac = _buildCirc(c, ci)
    else:
        raise ValueError('Invalid boundary condition.')

    return Ar, Ac


# ---------------------------------------------------------------------------
# gcv_tik.m, gcv_tsvd.m, gcv_gtik.m
# ---------------------------------------------------------------------------

def gcv_tik(s, bhat):
    """GCV parameter choice for Tikhonov (FFT/DCT) image deblurring."""
    s = np.asarray(s).ravel()
    bhat = np.asarray(bhat).ravel()

    def G(alpha):
        phi_d = 1.0 / (np.abs(s) ** 2 + alpha ** 2)
        return np.sum(np.abs(bhat * phi_d) ** 2) / (np.sum(phi_d) ** 2)

    lo, hi = np.abs(s).min(), np.abs(s).max()
    res = minimize_scalar(G, bounds=(lo, hi), method='bounded')
    return res.x


def gcv_gtik(sa, sd, bhat):
    """GCV parameter choice for generalized Tikhonov (gtik_fft)."""
    sa = np.asarray(sa).ravel()
    sd = np.asarray(sd).ravel()
    bhat = np.asarray(bhat).ravel()

    def G(alpha):
        denom = np.abs(sa) ** 2 + alpha ** 2 * np.abs(sd) ** 2
        phi_d = np.abs(sd) ** 2 / denom
        return np.sum(np.abs(bhat * phi_d) ** 2) / (np.sum(phi_d) ** 2)

    lo, hi = np.abs(sa).min(), np.abs(sa).max()
    res = minimize_scalar(G, bounds=(lo, hi), method='bounded')
    return res.x


def gcv_tsvd(s, bhat):
    """GCV truncation-parameter choice for TSVD image deblurring."""
    s = np.asarray(s).ravel()
    bhat = np.asarray(bhat).ravel()

    order = np.argsort(np.abs(s))[::-1]        # descending order of |s|
    s_sorted = np.abs(s)[order]
    bhat_sorted = np.abs(bhat[order])
    n = len(s_sorted)

    bhat2 = bhat_sorted ** 2
    suffix = np.cumsum(bhat2[::-1])[::-1]      # suffix[i] = sum_{j>=i} bhat2[j]
    rho = suffix[1:n]                          # length n-1

    denom = (n - np.arange(1, n)) ** 2         # (n-1)^2, (n-2)^2, ..., 1^2
    G = rho / denom

    # Guard against being fooled by pairs of equal singular values
    equal_pairs = s_sorted[:n - 2] == s_sorted[1:n - 1]
    G[:n - 2][equal_pairs] = np.inf

    reg_min = np.argmin(G)
    return s_sorted[reg_min]


# ---------------------------------------------------------------------------
# tik_fft.m
# ---------------------------------------------------------------------------

def tik_fft(B, PSF, center, alpha=None):
    """Tikhonov image deblurring using the FFT algorithm (periodic BC)."""
    ci, cj = center
    S = np.fft.fft2(np.roll(PSF, (-ci, -cj), axis=(0, 1)))
    s = S.ravel(order='F')

    bhat = np.fft.fft2(B)
    bhat_v = bhat.ravel(order='F')

    if alpha is None:
        alpha = gcv_tik(s, bhat_v)

    D = np.conj(s) * s + abs(alpha) ** 2
    bhat_v = np.conj(s) * bhat_v
    xhat = (bhat_v / D).reshape(B.shape, order='F')
    X = np.real(np.fft.ifft2(xhat))
    return X, alpha


# ---------------------------------------------------------------------------
# tik_dct.m
# ---------------------------------------------------------------------------

def tik_dct(B, PSF, center, alpha=None):
    """Tikhonov image deblurring using the DCT algorithm (reflexive BC)."""
    e1 = np.zeros_like(PSF)
    e1[0, 0] = 1.0

    bhat = dct2(B)
    S = dct2(dctshift(PSF, center)) / dct2(e1)

    bhat_v = bhat.ravel(order='F')
    s = S.ravel(order='F')

    if alpha is None:
        alpha = gcv_tik(s, bhat_v)

    D = s ** 2 + abs(alpha) ** 2
    bhat_v = s * bhat_v
    xhat = (bhat_v / D).reshape(B.shape, order='F')
    X = idct2(xhat)
    return X, alpha


# ---------------------------------------------------------------------------
# tik_sep.m
# ---------------------------------------------------------------------------

def tik_sep(B, PSF, center, alpha=None, BC='zero'):
    """Tikhonov image deblurring using the Kronecker-product decomposition."""
    Ar, Ac = kronDecomp(PSF, center, BC)

    Ur, Sr, Vrt = svd(Ar)
    Uc, Sc, Vct = svd(Ac)
    Vr, Vc = Vrt.T, Vct.T

    bhat = Uc.T @ B @ Ur
    bhat_v = bhat.ravel(order='F')
    s = np.kron(Sr, Sc)

    if alpha is None:
        alpha = gcv_tik(s, bhat_v)

    D = np.abs(s) ** 2 + abs(alpha) ** 2
    bhat_v = s * bhat_v
    xhat = (bhat_v / D).reshape(B.shape, order='F')
    X = Vc @ xhat @ Vr.T
    return X, alpha


# ---------------------------------------------------------------------------
# gtik_fft.m
# ---------------------------------------------------------------------------

def gtik_fft(B, PSF, center, Pd=None, alpha=None):
    """
    Generalized Tikhonov image deblurring using the FFT algorithm, with a
    user-specified regularization-operator stencil Pd (e.g. a Laplacian).
    """
    if Pd is None:
        Pd = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    if Pd.shape != (3, 3):
        raise ValueError('Illegal stencil for regularization operator')
    center_d = (1, 1)  # 0-based center of the 3x3 stencil

    ci, cj = center
    bhat = np.fft.fft2(B)
    bhat_v = bhat.ravel(order='F')

    Sa = np.fft.fft2(np.roll(PSF, (-ci, -cj), axis=(0, 1)))
    sa = Sa.ravel(order='F')

    Pd_full = padPSF(Pd, B.shape)
    Sd = np.fft.fft2(np.roll(Pd_full, (-center_d[0], -center_d[1]), axis=(0, 1)))
    sd = Sd.ravel(order='F')

    if alpha is None:
        alpha = gcv_gtik(sa, sd, bhat_v)

    D = np.conj(sa) * sa + abs(alpha) ** 2 * np.conj(sd) * sd
    bhat_v = np.conj(sa) * bhat_v
    xhat = (bhat_v / D).reshape(B.shape, order='F')
    X = np.real(np.fft.ifft2(xhat))
    return X, alpha


# ---------------------------------------------------------------------------
# tsvd_fft.m
# ---------------------------------------------------------------------------

def tsvd_fft(B, PSF, center, tol=None):
    """Truncated SVD image deblurring using the FFT algorithm (periodic BC)."""
    ci, cj = center
    S = np.fft.fft2(np.roll(PSF, (-ci, -cj), axis=(0, 1)))
    bhat = np.fft.fft2(B)

    s_v = S.ravel(order='F')
    bhat_v = bhat.ravel(order='F')

    if tol is None:
        tol = gcv_tsvd(s_v, bhat_v)

    Phi = (np.abs(S) >= tol)
    Sfilt = np.zeros_like(S, dtype=complex)
    Sfilt[Phi] = Phi[Phi] / S[Phi]

    X = np.real(np.fft.ifft2(bhat * Sfilt))
    return X, tol


# ---------------------------------------------------------------------------
# tsvd_dct.m
# ---------------------------------------------------------------------------

def tsvd_dct(B, PSF, center, tol=None):
    """Truncated SVD image deblurring using the DCT algorithm (reflexive BC)."""
    e1 = np.zeros_like(PSF)
    e1[0, 0] = 1.0

    bhat = dct2(B)
    S = dct2(dctshift(PSF, center)) / dct2(e1)

    s_v = S.ravel(order='F')
    bhat_v = bhat.ravel(order='F')

    if tol is None:
        tol = gcv_tsvd(s_v, bhat_v)

    Phi = (np.abs(S) >= tol)
    Sfilt = np.zeros_like(S)
    Sfilt[Phi] = Phi[Phi] / S[Phi]

    X = idct2(bhat * Sfilt)
    return X, tol


# ---------------------------------------------------------------------------
# tsvd_sep.m
# ---------------------------------------------------------------------------

def tsvd_sep(B, PSF, center, tol=None, BC='zero'):
    """Truncated SVD image deblurring using the Kronecker decomposition."""
    Ar, Ac = kronDecomp(PSF, center, BC)

    Ur, Sr, Vrt = svd(Ar)
    Uc, Sc, Vct = svd(Ac)
    Vr, Vc = Vrt.T, Vct.T

    bhat = Uc.T @ B @ Ur
    bhat_v = bhat.ravel(order='F')
    s = np.kron(Sr, Sc)

    if tol is None:
        tol = gcv_tsvd(s, bhat_v)

    Phi = (np.abs(s) >= tol)
    Sfilt = np.zeros_like(s)
    Sfilt[Phi] = Phi[Phi] / s[Phi]

    Bhat = (bhat_v * Sfilt).reshape(B.shape, order='F')
    X = Vc @ Bhat @ Vr.T
    return X, tol

def gcv_gtik_dct(sa, sd, bhat):
    """
    GCV parameter choice for generalized Tikhonov with DCT.

    Parameters
    ----------
    sa : ndarray
        DCT-domain eigenvalues of the blurring matrix A.
    sd : ndarray
        DCT-domain eigenvalues of the regularization operator L.
    bhat : ndarray
        DCT coefficients of the blurred image.

    Returns
    -------
    alpha : float
        Optimal regularization parameter.
    """
    def GCV(alpha):
        denom = sa**2 + alpha**2 * sd**2
        # Protect against division by zero
        denom = np.where(denom == 0, np.finfo(float).eps, denom)
        phi_d = sd**2 / denom
        G = np.sum(np.abs(bhat * phi_d)**2) / (np.sum(phi_d)**2)
        return G

    # Search in a reasonable range
    sa_abs = np.abs(sa)
    result = minimize_scalar(
        GCV,
        bounds=(sa_abs.min(), sa_abs.max()),
        method='bounded'
    )
    return result.x


def gtik_dct(B, PSF, center, Pd=None, alpha=None):
    """
    Generalized Tikhonov image deblurring using the DCT algorithm
    (reflexive boundary conditions).

    Solves:
        min_x ||A x - b||^2 + alpha^2 ||L x||^2

    where A is the blurring matrix (reflexive BC) and L is a
    regularization operator defined by the stencil Pd.

    Parameters
    ----------
    B : ndarray (m, n)
        Blurred image.
    PSF : ndarray (m, n)
        Point spread function, same size as B.
    center : tuple (int, int)
        (row, col) indices of the PSF center (0-indexed).
    Pd : ndarray (3, 3), optional
        Stencil for the regularization operator L. Common choices:

        Identity (standard Tikhonov):
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]

        1st derivative (rows):
            [[0, 0, 0],
             [1,-1, 0],
             [0, 0, 0]]

        1st derivative (cols):
            [[0, 1, 0],
             [0,-1, 0],
             [0, 0, 0]]

        2nd derivative (rows):
            [[0, 0, 0],
             [1,-2, 1],
             [0, 0, 0]]

        2nd derivative (cols):
            [[0, 1, 0],
             [0,-2, 0],
             [0, 1, 0]]

        Laplacian (isotropic 2nd derivative):
            [[0, 1, 0],
             [1,-4, 1],
             [0, 1, 0]]

        Default: identity (reduces to standard Tikhonov).

    alpha : float, optional
        Regularization parameter. If None, chosen by GCV.

    Returns
    -------
    X : ndarray (m, n)
        Restored image.
    alpha : float
        Regularization parameter used.

    Notes
    -----
    The regularization operator L must be a convolution with reflexive
    boundary conditions for the DCT diagonalization to apply. This is
    satisfied by any symmetric stencil (Laplacian, 2nd derivatives).

    For non-symmetric stencils (1st derivatives), the resulting matrix
    is not symmetric, and the DCT diagonalization is only approximate.
    In practice, 1st derivative stencils still work reasonably well
    because the asymmetry is mild.

    Reference: Chapter 7 of Hansen, Nagy, O'Leary (2006), adapted for
    reflexive boundary conditions.
    """
    # Default regularization operator: identity
    if Pd is None:
        Pd = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]], dtype=float)

    if Pd.shape != (3, 3):
        raise ValueError("Regularization stencil must be 3x3")

    # Center of the 3x3 stencil (0-indexed)
    center_d = (1, 1)

    # ── Step 1: Compute DCT-domain eigenvalues of A ──
    # (same as in tik_dct)
    e1 = np.zeros_like(PSF)
    e1[0, 0] = 1.0

    bhat = dctn(B, type=2, norm='ortho')
    Sa = dctn(dctshift(PSF, center), type=2, norm='ortho') / dctn(e1, type=2, norm='ortho')

    # ── Step 2: Compute DCT-domain eigenvalues of L ──
    # Pad the 3x3 stencil to image size, then apply dctshift
    Pd_padded = np.zeros_like(PSF)
    Pd_padded[:3, :3] = Pd
    Sd = dctn(dctshift(Pd_padded, center_d), type=2, norm='ortho') / dctn(e1, type=2, norm='ortho')

    # Flatten for element-wise operations
    bhat_flat = bhat.ravel()
    sa = Sa.ravel()
    sd = Sd.ravel()

    # ── Step 3: Choose alpha via GCV if not provided ──
    if alpha is None:
        alpha = gcv_gtik_dct(sa, sd, bhat_flat)

    # ── Step 4: Compute the regularized solution in DCT domain ──
    # Filter: xhat = sa * bhat / (sa^2 + alpha^2 * sd^2)
    denom = sa**2 + alpha**2 * sd**2
    # Protect against division by zero (can happen if both sa and sd are zero)
    denom = np.where(denom == 0, np.finfo(float).eps, denom)

    xhat_flat = (sa * bhat_flat) / denom
    xhat = xhat_flat.reshape(B.shape)

    # ── Step 5: Inverse DCT to get spatial-domain solution ──
    X = idctn(xhat, type=2, norm='ortho')

    return X, alpha
