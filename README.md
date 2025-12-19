# DP-EGGROLL

Differentially Private Evolution Strategies via Fitness Vector Privatization

![DP-EGGROLL Banner](banner.png)

## Contents

- `dp_eggroll.py` — Implementation of DP-EGGROLL with RDP accounting, Poisson subsampling, and centered clipping
- `dp_eggroll.ipynb` — Jupyter notebook with experimental results comparing DP-EGGROLL vs DP-SGD on MNIST

**Coming soon:** LaTeX paper and Medium article

## Usage

```bash
python dp_eggroll.py
```

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
opacus>=1.4.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

Install via:
```bash
pip install torch torchvision opacus numpy pandas matplotlib
```

Developed and tested on Google Colab with CUDA 12.6.

## Author

David Zagardo  
dave@greenwillowstudios.com