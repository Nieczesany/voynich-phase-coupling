<img width="1424" height="752" alt="image_44544d9e" src="https://github.com/user-attachments/assets/b08fdb02-84ba-421a-8056-222b72d959cb" />

# Voynich MS 408: Analog-Computer Phase-Coupling Test (Python)

A falsifiable, geometry-first test for the Voynich Manuscript (MS 408). We treat astronomy/astrology folios as paper-based gears (volvelles). If that’s true, adjacent concentric rings should exhibit non-random angular phase coupling.

## Core idea
- **Language-agnostic:** Test mechanical coupling before any linguistic claims.
- **Pipeline:** Polar unwrapping (OpenCV), 1D angular ink-density signals, circular FFT cross-correlation, permutation-based nulls, FDR correction.
- **Outcome:** Either robust, reproducible coupling exists (evidence for “paper mechanism”), or the hypothesis weakens.

## Method (pipeline)
1. **Preprocessing:** CLAHE + Sauvola (ink over parchment artifacts).
2. **Geometry:** OpenCV `warpPolar` &rarr; unwrap concentric rings into linear angular space.
3. **Signal:** Per-ring 1D angular ink-density profiles; z-score; circular smoothing.
4. **Coupling:** Circular cross-correlation (FFT) across 0–360° &rarr; \(k_{max}\) (peak angle) + effect size.
5. **Statistics:**
   - Permutation null (10k angular shuffles) for max-corr detection.
   - Benjamini–Hochberg FDR for multiple ring pairs.
   - Optional fixed-angle test with rotation-based null for a priori astronomical angles.

## Install
- Python &ge; 3.10
- `pip install -r requirements.txt`

## Quickstart
1. Place your Beinecke images locally (we do not redistribute them).
2. Edit `configs/vms_astronomy.yaml`: set image paths, center [x, y], ring radii.
3. Run:
   ```bash
   python voynich_phase_coupling.py --config configs/vms_astronomy.yaml --n-iters 10000 --n-bins 360 --save-dir outputs/run1
   ```

## Pre-registrable success criteria
- FDR-adjusted $p < 0.01$ for &ge; 70% adjacent ring pairs across &ge; 3 folios.
- Coherent $k_{max}$ angles within a folio (low variance).
- Robustness to small perturbations: center (&plusmn;3 px), rings (&plusmn;5 px), binning (360 vs 720).

## Reproducibility safeguards
- Permutation nulls tailored to angular sampling.
- FDR correction; fixed preprocessing across folios.
- Sensitivity analysis logged with parameters and seeds.

## Ethics and data
No redistribution of Beinecke TIFFs. Users supply local paths per library policy. This repo provides code, configs, and synthetic demos only.

## License
MIT — see LICENSE.
