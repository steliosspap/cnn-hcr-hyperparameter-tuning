# Methodology Notes: CNN Hyperparameter Tuning for EMNIST

## Dataset: EMNIST/byclass

- **Source**: TensorFlow Datasets (TFDS)
- **Content**: Extended MNIST dataset with 62 character classes (0-9, A-Z, a-z)
- **Format**: 28×28 grayscale images, normalized to [0,1] range
- **Preprocessing**: 
  - Cast to float32, normalize by 255
  - Expand dimensions to (28,28,1) for CNN input
  - One-hot encode labels for 62 classes
- **Data Pipeline**: 
  - Train: cache → shuffle(10k) → batch → prefetch → map(preprocess)
  - Test: map(preprocess) → batch → prefetch

## Model Architecture

Replicates the Extended Essay Appendix architecture:

```
Input(28,28,1) 
→ Conv2D(k, 3×3, ReLU)           # k filters
→ Conv2D(2k, 3×3, ReLU, same)    # 2k filters, same padding
→ MaxPool2D(2×2, stride=2)       # Reduce spatial dimensions
→ Conv2D(4k, 3×3, ReLU, valid)   # 4k filters, valid padding
→ MaxPool2D(2×2, stride=2)       # Further reduce dimensions
→ Flatten()                       # Convert to 1D
→ Dense(256, ReLU)               # Fully connected layer
→ Dense(512, ReLU)               # Fully connected layer
→ Dropout(0.5)                   # Regularization
→ Dense(62, softmax)             # Output layer for 62 classes
```

**Optimizer**: Adamax with configurable learning rate
**Loss**: Categorical crossentropy (matches essay specification)
**Metrics**: Accuracy

## Hyperparameter Sweep

### Learning Rates
- **Values**: [1e-4, 1e-3, 1e-2, 1e-1]
- **Rationale**: Cover 4 orders of magnitude to find optimal convergence
- **Expected**: Very low (1e-4) may be too slow, very high (1e-1) may diverge

### Kernel Counts
- **Values**: [16, 32, 64, 128]
- **Rationale**: Test model capacity vs. overfitting trade-off
- **Expected**: Higher counts increase model capacity but risk overfitting

### Epochs
- **Values**: [2, 5, 10]
- **Rationale**: Balance training time vs. convergence
- **Expected**: More epochs generally improve accuracy but increase training time

## Experimental Design

### Grid Search Strategy
- **Quick Grid** (default): 2×2×2 = 8 combinations
- **Full Grid**: 4×4×3 = 48 combinations
- **Reproducibility**: Fixed seeds for TF, NumPy, Python, and random

### Performance Considerations
- **Batch Size**: Default 128 (configurable)
- **Early Stopping**: Optional with configurable patience
- **Memory Management**: Clear Keras backend between runs
- **Data Efficiency**: Cache, shuffle, and prefetch for optimal throughput

## Trade-offs and Observations

### Accuracy vs. Compute
- **Higher kernel counts**: Better feature extraction but increased memory/compute
- **More epochs**: Better convergence but longer training time
- **Learning rate**: Critical for convergence speed and final accuracy

### Generalization vs. Memorization
- **Too many kernels (128)**: Risk of overfitting to training data
- **Too few epochs**: May not reach optimal weights
- **Dropout (0.5)**: Helps prevent overfitting in dense layers

### Expected Findings
Based on the Extended Essay results:
- **Optimal configuration**: ~64 kernels + moderate LR (~1e-3) + 10 epochs
- **128 kernels**: May show overfitting (high train acc, lower val acc)
- **Learning rate sweet spot**: Likely between 1e-3 and 1e-2
- **Epochs**: 10 typically provides best validation accuracy

## Reproducibility Features

- **Seed setting**: All random sources controlled
- **Data pipeline**: Consistent preprocessing and augmentation
- **Model building**: Deterministic architecture construction
- **Results logging**: Comprehensive CSV output with all parameters
- **Visualization**: Standardized plotting for easy comparison
