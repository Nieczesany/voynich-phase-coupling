# Voynich MS 408: Analog-Computer Phase-Coupling Test (Python)

A falsifiable, geometry-first test for the Voynich Manuscript (MS 408). We treat astronomy/astrology folios as paper-based gears (volvelles). If that’s true, adjacent concentric rings should exhibit non-random angular phase coupling.

## Core Idea
- **Geometry-First & Language-Agnostic:** Test mechanical and physical coupling before making arbitrary linguistic claims.
- **Pipeline:** Preprocessing (Sauvola), Polar unwrapping (`linearPolar`), 1D angular ink-density profiles, circular FFT cross-correlation, permutation-based nulls, FDR correction, and cryptographic checksum validation via Gematria.
- **Outcome:** Either robust, reproducible coupling and valid checksums exist (proving a "paper mechanism"), or the hypothesis weakens deterministically.

## Method (Pipeline)
1. **Preprocessing (Pillar 1):** CLAHE + Sauvola local adaptive thresholding to eliminate 500-year-old parchment noise, artifacts, and historical stains, isolating sharp ink.
2. **Geometry (Pillar 2):** OpenCV `linearPolar` mapping to unwrap concentric rings into flat, linear angular spaces (360 bins = 1° resolution).
3. **Signal Extraction:** Per-ring 1D angular ink-density profiles; z-score normalization; circular smoothing.
4. **Coupling Analysis:** Circular cross-correlation via FFT across 0–360° to isolate the exact phase peak (\(k_{max}\)) and effect size.
5. **Linguistic Checksum (Pillar 3 & 4):** Algorithmic NLP mapping of Extensible Voynich Alphabet (EVA) tokens to Ancient Hebrew consonantal roots. Calculation of mathematical check values (Gematria) to match targets (e.g., 53 for Sun, 713 for Saturn) and filter random noise.
6. **Statistics:**
   - Permutation null (10k angular shuffles) for max-corr detection.
   - Benjamini–Hochberg FDR for multiple ring pairs.
   - Optional fixed-angle test with rotation-based null for a priori astronomical angles (e.g., Saros Cycle).

## Install
- Python \(\ge\) 3.10
- `pip install numpy opencv-python scikit-image scipy statsmodels matplotlib pyyaml`

## Quickstart
1. Place your Beinecke images locally (we do not redistribute them).
2. Edit `configs/vms_astronomy.yaml`: set image paths, center [x, y], ring radii, and target shifts.
3. Run:
   ```bash
   python voynich_phase_coupling.py --config configs/vms_astronomy.yaml --n-iters 10000 --n-bins 360 --save-dir outputs/run1
   ```

## Pre-registrable Success Criteria
- FDR-adjusted $p < 0.01$ for $\ge$ 70% adjacent ring pairs across $\ge$ 3 folios.
- Coherent $k_{max}$ angles matching specific cryptographic/astronomical constants.
- Gematria checksum success for exposed points, ruling out random statistical drift.
- Robustness to small perturbations: center ($\pm$3 px), rings ($\pm$5 px), binning (360 vs 720).

## Reproducibility Safeguards
- Permutation nulls tailored strictly to angular circular sampling.
- FDR correction; fixed preprocessing across folios.
- Sensitivity analysis logged with deterministic parameters and random seeds.

## Ethics and Data
No redistribution of Beinecke rare library TIFF scans. Users supply local paths per library policy. This repo provides code, configs, and synthetic validation benchmarks only.

## License
MIT — see LICENSE for details.
