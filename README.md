# CNN Hyperparameter Tuning for Handwritten Character Recognition (EMNIST)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional, reproducible investigation into how kernel count, learning rate, and epochs affect CNN accuracy for handwritten character recognition using the EMNIST byclass dataset.

## Research Question

**How do convolutional neural network hyperparameters (kernel count, learning rate, and training epochs) impact validation accuracy for handwritten character recognition on the EMNIST byclass dataset?**

This project replicates and extends the hyperparameter analysis from the IB Extended Essay (Grade A) to provide a comprehensive understanding of CNN optimization for character recognition tasks.

## TL;DR Findings

Based on the Extended Essay results and expected outcomes:

- **64 kernels + moderate learning rate (~1e-3) + 10 epochs** typically yields the best validation accuracy
- **128 kernels** may lead to overfitting despite higher training accuracy
- **Learning rate sweet spot** appears between 1e-3 and 1e-2 for stable convergence
- **10 epochs** generally provides optimal validation performance, though 5 epochs can be sufficient for quick iterations
- **Early stopping** can help prevent overfitting when using higher kernel counts

## Quickstart

### Prerequisites
- Python 3.10+
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/steliosspap/cnn-hcr-hyperparameter-tuning.git
cd cnn-hcr-hyperparameter-tuning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

**Quick grid (recommended for first run):**
```bash
python src/cnn_hcr_experiment.py
```
- Learning rates: [1e-3, 1e-2]
- Kernel counts: [32, 64]
- Epochs: [2, 5]
- Total combinations: 8

**Full grid (comprehensive sweep):**
```bash
python src/cnn_hcr_experiment.py --full-grid
```
- Learning rates: [1e-4, 1e-3, 1e-2, 1e-1]
- Kernel counts: [16, 32, 64, 128]
- Epochs: [2, 5, 10]
- Total combinations: 48

**Additional options:**
```bash
# Custom batch size and early stopping
python src/cnn_hcr_experiment.py --batch-size 64 --patience 3

# Quick smoke test with limited data
python src/cnn_hcr_experiment.py --limit-train 1000 --limit-test 500
```

## Outputs

### Results CSV (`results/results.csv`)
Each row contains:
- `learning_rate`, `kernel_count`, `epochs`, `batch_size`
- `train_accuracy`, `val_accuracy` (final epoch values)
- Sorted by validation accuracy (best first)

### Visualization Plots (`results/`)
1. **`accuracy_vs_lr.png`** - Learning rate impact (log scale)
2. **`accuracy_vs_kernels.png`** - Kernel count impact
3. **`accuracy_vs_epochs.png`** - Training epochs impact

### Plot Interpretation

- **Learning Rate Plot**: Look for the "sweet spot" where accuracy peaks before declining
- **Kernel Count Plot**: Identify the point where more kernels stop improving validation accuracy (overfitting threshold)
- **Epochs Plot**: Find the optimal training duration for your use case

## Model Architecture

Replicates the Extended Essay Appendix architecture:
```
Input(28,28,1) → Conv2D(k) → Conv2D(2k, same) → MaxPool → Conv2D(4k) → MaxPool → Flatten → Dense(256) → Dense(512) → Dropout(0.5) → Dense(62, softmax)
```

- **Optimizer**: Adamax
- **Loss**: Categorical crossentropy
- **Dataset**: EMNIST/byclass (62 classes: 0-9, A-Z, a-z)
- **Preprocessing**: Normalize to [0,1], expand to (28,28,1), one-hot labels

## Extended Essay

**Note**: This repository is designed to work with the IB Extended Essay document. Please place your `Extended_Essay.pdf` file in the `docs/` directory to maintain the complete research context.

The Extended Essay (Grade A) provides the theoretical foundation and initial experimental results that this codebase reproduces and extends.

## Methodology

See `docs/methodology_notes.md` for detailed information about:
- Dataset characteristics and preprocessing
- Model architecture rationale
- Hyperparameter selection strategy
- Experimental design considerations
- Expected trade-offs and findings

## Reproducibility

- **Seeds**: All random sources controlled (TF, NumPy, Python, random)
- **Data Pipeline**: Consistent preprocessing and augmentation
- **Model Building**: Deterministic architecture construction
- **Results Logging**: Comprehensive CSV output with all parameters

## Performance Tips

- **Quick iterations**: Use `--limit-train 1000` for faster development
- **Memory management**: Lower batch sizes if encountering OOM errors
- **Early stopping**: Use `--patience 3` to prevent overfitting
- **Full sweep**: Run overnight or on cloud compute for comprehensive results

## Contributing

This is a research reproduction project. If you find issues or have suggestions:
1. Check the methodology notes for context
2. Ensure reproducibility with the provided seeds
3. Consider the Extended Essay findings when interpreting results

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:
```
@software{spapadopoulos2024cnnhcr,
  title={CNN Hyperparameter Tuning for Handwritten Character Recognition},
  author={Stelios Spapadopoulos},
  year={2024},
  url={https://github.com/steliosspap/cnn-hcr-hyperparameter-tuning}
}
```

---

**Built with ❤️ for reproducible ML research**
