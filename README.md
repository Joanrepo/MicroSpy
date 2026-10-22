# MicroSpy: Stealing fine-grained architecture of optimized DNNs via physical side-channel analysis

This is the paper repository for MicroSpy, which aims to steal the fine-grained architectures of optimized DNNs via physical side-channel analysis.

---

## Dependencies

Python environment recommended: **Python 3.8+** (a recent 3.x version should work).

Key Python packages used by the provided scripts include:
- `torch`
- `numpy`
- `h5py`
- `matplotlib`
- `tqdm`
- `scipy`
- `pywt` (`PyWavelets`)

---

## Code Structure

- **`src/LA-Trans-train.py`**  
  Training and evaluation script for the proposed Layer-Aware Transformer (LA-Trans) model in MicroSpy.

- **`src/MLP-backbone-alt-train.py`**  
  Training and evaluation script for the MLP-backbone baseline.

- **`src/CNN-backbone-alt-train.py`**  
  Training and evaluation script for the CNN-backbone baseline.

- **`src/RNN-backbone-alt-train.py`**  
  Training and evaluation script for the RNN-backbone baseline.