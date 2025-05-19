
# Bird Call CBIR

## Description  
A lightweight Python framework for content-based retrieval of bird-call recordings. It lets you extract spectrogram‐based features from audio, build similarity matrices, and evaluate retrieval performance (precision/recall).

## Experimental Findings (Semi-Exhaustive Testing)

Our semi-exhaustive sweep over resampling methods and spectrogram parameters yielded the following insights:

![image](https://github.com/user-attachments/assets/e6db8cce-2cac-4f55-b962-7e982c891540)

![image](https://github.com/user-attachments/assets/471ab84b-d4d0-4123-a680-bb20d77f00f2)

## Experimental Findings (Semi-Exhaustive Testing)

> **Note:** Whenever the “resampling method” is listed as **none**, no downsampling was actually performed—the data remained at the original **44.1 kHz** rate, despite a nominal target of 22.05 kHz.

1. **Baseline (no resampling, 44.1 kHz)**

   * Mean precision ≈ 0.662; mean recall ≈ 0.574. This serves as our reference point.

2. **Librosa Resampling (proper anti-alias filtering to 22.05 kHz)**

   * Precision ≈ 0.649 (−1.9% vs. baseline)
   * Recall ≈ 0.560 (−2.4% vs. baseline)
   * **Insight:** Cleaner spectrum but slightly lower retrieval accuracy in our CBIR setup.

3. **Window & Spectrogram Parameters**

   * **Window size = 512 samples** (regardless of sample rate) consistently outperformed larger windows by ≈ 4%.
   * **Overlap ≈ 20%** was optimal; zero overlap was poorest.
   * **Blackman** (or **Hann**) windows gave slightly more stable results than Hamming.

4. **Spectrogram Mode & Scaling**

   * **Magnitude**-only spectrograms beat PSD by ≈ 3.7%.
   * Within magnitude mode, **spectrum** scaling edged **density** by ≈ 0.7%.

## Folder Structure  

```
.
├── src/                                     # Core CBIR code (feature extraction, matching)
├── Audio Resampling for Bird Call CBIR.md
├── Audio Resampling for Bird Call CBIR.pdf  # Resampling experiment report
├── Plot\_precision\_recall.ipynb            # Notebook to plot PR curves
├── experiment\_results.csv                  # CSV of precision/recall scores
├── generate\_matrix.py                      # Builds similarity matrices
├── workflow\.yaml                           # Argo Workflow definition for test runs
├── Dockerfile                               # Container image spec
├── .dockerignore
└── .gitignore
````

## Requirements  
- Python 3.8+  
- librosa, NumPy, SciPy, scikit-learn, matplotlib (for plotting)  
- (Optional) Kubernetes + Argo Workflows for large-scale testing  

## Installation  
```bash
git clone https://github.com/michael-pisman/bird_call_cbir.git
cd bird_call_cbir
pip install -r src/requirements.txt
````

## Docker

```

## Usage

1. **Feature extraction**

   ```bash
   python src/extract_features.py --input path/to/audio/dir --output features.pkl
   ```
2. **Similarity matrix & evaluation**

   ```bash
   python generate_matrix.py --features features.pkl --out experiment_results.csv
   ```
3. **Plot results**
   Open `Plot_precision_recall.ipynb` in Jupyter and run all cells.

## Argo Workflows & Test Bed

To automate large-scale experiments (different resampling rates, filter designs, feature types), we use an Argo Workflow defined in `workflow.yaml`. This can run on any Kubernetes cluster with Argo installed, spin up pods for each parameter set, collect results in a shared volume, and aggregate metrics automatically. Just apply:

```bash
kubectl apply -f workflow.yaml
```

