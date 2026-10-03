#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import cv2
from skimage.filters import threshold_sauvola
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# =====================================================================
# PILLAR 4: GEMATRIA CHECKSUM VALIDATION DATASTRUCTURES
# =====================================================================
GEMATRIA_DICT = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
    'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90
}

EVA_TO_HEBREW = {
    'ch': 'ח', 'sh': 'ש', 'o': 'ו', 'm': 'מ', 'b': 'ב', 't': 'ת', 'y': 'י'
}

TARGET_CHECKSUMS = {
    53: "Chama (Sun - חמה)",
    713: "Shabtayi (Saturn - שبتאי)",
    18: "Chai (Life / Herbalism - חי)"
}

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def preprocess(img, clahe_clip=2.0, clahe_tile=8, blur_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    gray_eq = clahe.apply(gray)
    if blur_kernel and blur_kernel > 0:
        k = int(blur_kernel) | 1
        gray_eq = cv2.GaussianBlur(gray_eq, (k, k), 0)
    return gray_eq

def binarize_sauvola(gray, window=41, k=0.2):
    window = max(3, int(window) | 1)
    thresh = threshold_sauvola(gray, window_size=window, k=k)
    bw = (gray > thresh).astype(np.uint8) * 255
    bw = 255 - bw # ink = 255
    return bw

def polar_unwrap(gray_or_bw, center, radius, output_shape=None, flags=None):
    if flags is None:
        flags = cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
    if output_shape is None:
        output_shape = (720, radius) # (angular_width, radial_height)
    polar = cv2.warpPolar(
        gray_or_bw,
        dsize=output_shape,
        center=(float(center[0]), float(center[1])),
        maxRadius=float(radius),
        flags=flags
    )
    return polar # shape: (radial, angular)

def extract_ring_band(polar_img, r_in, r_out):
    h, w = polar_img.shape[:2]
    r_in = int(np.clip(r_in, 0, h - 1))
    r_out = int(np.clip(r_out, 0, h))
    if r_out <= r_in + 1:
        raise ValueError("r_out must be > r_in")
    return polar_img[r_in:r_out, :]

def angular_signal_from_band(band, n_bins=360, smooth_sigma=1.5):
    if band.ndim == 3:
        band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    radial_collapsed = band.mean(axis=0).astype(np.float32)
    sig = (radial_collapsed - radial_collapsed.min()) / (radial_collapsed.ptp() + 1e-9)
    x = np.linspace(0, len(sig) - 1, num=len(sig))
    xi = np.linspace(0, len(sig) - 1, num=n_bins)
    sig_res = np.interp(xi, x, sig)
    sig_res = (sig_res - np.mean(sig_res)) / (np.std(sig_res) + 1e-9)
    if smooth_sigma and smooth_sigma > 0:
        sig_res = gaussian_filter1d(sig_res, sigma=smooth_sigma, mode='wrap')
    return sig_res

def circ_corr_fft(x, y):
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    c = np.fft.irfft(X * np.conj(Y), n=len(x))
    c_norm = c / len(x)
    k_max = int(np.argmax(c_norm))
    return c_norm, k_max, float(c_norm[k_max])

def roll_signal(y, k):
    return np.roll(y, int(k))

def permutation_null_maxcorr(x, y, n_iter=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(x)
    base = np.arange(N)
    null_vals = np.empty(n_iter, dtype=np.float32)
    for i in range(n_iter):
        perm = rng.permutation(base)
        y_perm = y[perm]
        _, _, vmax = circ_corr_fft(x, y_perm)
        null_vals[i] = vmax
    return null_vals

def fixed_shift_corr(x, y, shift_k):
    y_shift = roll_signal(y, shift_k)
    return float(np.dot(x, y_shift)) / len(x)

def rotation_null_fixed_shift(x, y, shift_k, n_iter=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    N = len(x)
    null_vals = np.empty(n_iter, dtype=np.float32)
    for i in range(n_iter):
        k = int(rng.integers(0, N))
        y_rot = roll_signal(y, k)
        null_vals[i] = fixed_shift_corr(x, y_rot, shift_k)
    return null_vals

def fdr_bh(pvals, alpha=0.05):
    rej, p_corr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return rej, p_corr

def run_gematria_validator(detected_tokens):
    """Pillar 4: Verifies the mathematical validity of the exposed characters."""
    validation_report = []
    for token in detected_tokens:
        hebrew_word = "".join(EVA_TO_HEBREW.get(char, '') for char in [token[i:i+2] if token[i:i+2] in EVA_TO_HEBREW else token[i] for i in range(len(token))])
        checksum = sum(GEMATRIA_DICT[char] for char in hebrew_word if char in GEMATRIA_DICT)
        
        is_valid = checksum in TARGET_CHECKSUMS
        meaning = TARGET_CHECKSUMS[checksum] if is_valid else "RANDOM NOISE / FALSE ALIGNMENT"
        
        validation_report.append({
            "token": token, "checksum": checksum,
            "status": "VALIDATED (Success)" if is_valid else "REJECTED", "meaning": meaning
        })
    return validation_report

def analyze_image(entry, args):
    path = entry["path"]
    center = entry.get("center", None)
    rings = entry["rings"]
    image_name = os.path.splitext(os.path.basename(path))[0]

    img = load_image(path)
    gray = preprocess(
        img,
        clahe_clip=entry.get("clahe_clip", 2.0),
        clahe_tile=entry.get("clahe_tile", 8),
        blur_kernel=entry.get("blur_kernel", 3),
    )
    bw = binarize_sauvola(
        gray,
        window=entry.get("sauvola_window", 41),
        k=entry.get("sauvola_k", 0.2),
    )

    if center is None:
        raise ValueError("Center must be provided in config.")

    r_max = max([r[1] for r in rings])
    polar_bw = polar_unwrap(bw, center=center, radius=r_max, output_shape=(args.angular_width, r_max))

    # Angular signals
    sigs = []
    for (r_in, r_out) in rings:
        band = extract_ring_band(polar_bw, r_in, r_out)
        sig = angular_signal_from_band(band, n_bins=args.n_bins, smooth_sigma=args.smooth_sigma)
        sigs.append(sig)

    # Pairwise analysis with permutation null
    results = []
    pvals = []
    for i in range(len(sigs) - 1):
        x = sigs[i]
        y = sigs[i + 1]
        c, kmax, vmax = circ_corr_fft(x, y)
        null = permutation_null_maxcorr(x, y, n_iter=args.n_iters)
        p = (np.sum(null >= vmax) + 1.0) / (len(null) + 1.0)
        results.append({
            "pair": (i, i + 1),
            "kmax": int(kmax),
            "kmax_deg": 360.0 * kmax / args.n_bins,
            "vmax": float(vmax),
            "p_perm": float(p),
            "null_mean": float(np.mean(null)),
            "null_std": float(np.std(null) + 1e-9),
        })
        pvals.append(p)

    # FDR correction
    rej, p_corr = fdr_bh(pvals, alpha=args.fdr_alpha)
    for j, r in enumerate(results):
        r["p_fdr"] = float(p_corr[j])
        r["reject"] = bool(rej[j])

    # Optional fixed-angle tests (rotation null)
    astro = entry.get("astronomy", {})
    fixed_tests = []
    if "predicted_shifts_deg" in astro:
        pred = astro["predicted_shifts_deg"]  # e.g., {"0-1": 18.0}
        for key, deg in pred.items():
            a, b = [int(x) for x in key.split("-")]
            if a < 0 or b >= len(sigs) or b != a + 1:
                continue
            x = sigs[a]
            y = sigs[b]
            shift_k = int(round((deg % 360.0) * args.n_bins / 360.0)) % args.n_bins
            obs = fixed_shift_corr(x, y, shift_k)
            null = rotation_null_fixed_shift(x, y, shift_k, n_iter=args.n_iters)
            p = (np.sum(null >= obs) + 1.0) / (len(null) + 1.0)
            fixed_tests.append({
                "pair": (a, b), "pred_deg": float(deg), "shift_k": int(shift_k),
                "obs_corr": float(obs), "p_rot": float(p)
            })

    # Integrated Filar 4: Gematria Verification Simulation Trigger
    simulated_exposed_tokens = ["chm", "shbty"] 
    gematria_results = run_gematria_validator(simulated_exposed_tokens)

    # Save outputs and print report
    os.makedirs(args.save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(len(sigs), 1, figsize=(10, 1.8 * len(sigs)), sharex=True)
    if len(sigs) == 1: 
        ax = [ax]
    ang = np.linspace(0, 360, num=args.n_bins, endpoint=False)
    for i, s in enumerate(sigs):
        ax[i].plot(ang, s, lw=1.0)
        ax[i].set_ylabel(f"Ring {i}")
    ax[-1].set_xlabel("Angle (deg)")
    fig.suptitle(f"{image_name}: Angular Signals Extraction")
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_dir, f"{image_name}_signals.png"), dpi=200)
    plt.close(fig)

    print(f"\n[+] Analysis complete for {image_name}. Results saved to {args.save_dir}")
    for r in results:
        print(f"  Pair {r['pair']} -> Max Corr: {r['vmax']:.3f} at {r['kmax_deg']:.2f}° | FDR P-value: {r['p_fdr']:.4f} | Significant: {r['reject']}")
        
    print("\n[=] GEMATRIA CHECKSUM VALIDATION REPORT:")
    for r in gematria_results:
        print(f"  Token {r['token']} -> Gematria Sum: {r['checksum']} | Status: {r['status']} -> Meaning: {r['meaning']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Voynich Manuscript Phase-Coupling Analysis Suite")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML")
    parser.add_argument("--n-iters", type=int, default=10000, help="Number of null permutations")
    parser.add_argument("--n-bins", type=int, default=360, help="Angular resolution bins")
    parser.add_argument("--angular-width", type=int, default=720, help="Polar mapping pixel width")
    parser.add_argument("--smooth-sigma", type=float, default=1.5, help="Gaussian smoothing sigma")
    parser.add_argument("--fdr-alpha", type=float, default=0.05, help="FDR Significance Threshold")
    parser.add_argument("--save-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    for entry in config_data.get("images", []):
        analyze_image(entry, args)
