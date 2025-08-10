#!/usr/bin/env python3
"""
CNN Hyperparameter Tuning for Handwritten Character Recognition
(EMNIST byclass)

Replicates the Extended Essay experiment:
- Sweeps kernel count, learning rate, and epochs
- Uses Keras (TF2) with Adamax optimizer
- Saves results CSV and 3 plots

Quick run (small grid):
    python src/cnn_hcr_experiment.py

Full run (full grid from the essay spec):
    python src/cnn_hcr_experiment.py --full-grid

Author: Stelios Spapadopoulos
License: MIT
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

# -----------------------
# Reproducibility helpers
# -----------------------

def set_all_seeds(seed: int = 42):
    """Set seeds for all random number generators for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

# -----------------------
# Data loading / pipeline
# -----------------------

NUM_CLASSES = 62
IMG_SIZE = (28, 28)

def _preprocess(image, label):
    """Preprocess EMNIST images and labels."""
    # image: uint8 (28,28), label: int
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    image = tf.expand_dims(image, axis=-1)  # (28,28,1)
    label = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
    return image, label

def build_datasets(batch_size: int = 128, limit_train: int = 0,
                   limit_test: int = 0):
    """
    Loads EMNIST byclass from TFDS, returns (train_ds, test_ds).
    Optional limits for quick smoke tests.
    """
    (train_ds, test_ds), info = tfds.load(
        "emnist/byclass",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        try_gcs=True,
    )
    
    if limit_train > 0:
        train_ds = train_ds.take(limit_train)
    if limit_test > 0:
        test_ds = test_ds.take(limit_test)
    
    train_ds = (
        train_ds
        .cache()
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
        .map(_preprocess, num_parallel_calls=AUTOTUNE)
    )
    
    test_ds = (
        test_ds
        .map(_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    
    return train_ds, test_ds

# -----------------------
# Model
# -----------------------

def build_model(kernel_count: int, learning_rate: float) -> tf.keras.Model:
    """
    Architecture matching the essay's Appendix:
    Conv2D(k) -> Conv2D(2k, same) -> MaxPool
    -> Conv2D(4k, valid) -> MaxPool
    -> Flatten -> Dense(256) -> Dense(512) -> Dropout(0.5) -> Dense(62, softmax)
    
    Optimizer: Adamax(lr)
    """
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    
    x = tf.keras.layers.Conv2D(filters=kernel_count, kernel_size=(3, 3), 
                               activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(filters=kernel_count * 2, kernel_size=(3, 3), 
                               activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    
    x = tf.keras.layers.Conv2D(filters=kernel_count * 4, kernel_size=(3, 3), 
                               activation='relu', padding='valid')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# -----------------------
# Experiment runner
# -----------------------

def run_grid(
    train_ds,
    test_ds,
    learning_rates: List[float],
    kernel_counts: List[int],
    epochs_list: List[int],
    batch_size: int,
    patience: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the hyperparameter grid search and return results DataFrame."""
    records = []
    callbacks = []
    
    if patience and patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=patience, 
            restore_best_weights=True
        ))
    
    for lr in learning_rates:
        for kc in kernel_counts:
            for ep in epochs_list:
                tf.keras.backend.clear_session()
                set_all_seeds(seed)  # reset before each run
                
                model = build_model(kernel_count=kc, learning_rate=lr)
                
                history = model.fit(
                    train_ds,
                    epochs=ep,
                    validation_data=test_ds,
                    verbose=0,
                    callbacks=callbacks
                )
                
                train_acc = float(history.history['accuracy'][-1])
                val_acc = float(history.history['val_accuracy'][-1])
                
                records.append(
                    dict(
                        learning_rate=lr,
                        kernel_count=kc,
                        epochs=ep,
                        batch_size=batch_size,
                        train_accuracy=train_acc,
                        val_accuracy=val_acc,
                    )
                )
                
                print(f"[DONE] lr={lr:<7} kernels={kc:<3} epochs={ep:<2} "
                      f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    
    df = pd.DataFrame.from_records(records).sort_values(
        by=['val_accuracy'], ascending=False
    )
    return df

# -----------------------
# Plotting
# -----------------------

def make_plots(df: pd.DataFrame, out_dir: Path):
    """Generate the three required plots and save them to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Effect of Learning Rate on Accuracy (group by kernels & epochs)
    plt.figure(figsize=(8, 5))
    for kc in sorted(df['kernel_count'].unique()):
        for ep in sorted(df['epochs'].unique()):
            subset = df[(df['kernel_count'] == kc) & (df['epochs'] == ep)]
            plt.plot(subset['learning_rate'], subset['val_accuracy'], 
                     marker='o', label=f'K={kc}, E={ep}')
    
    plt.xscale('log')
    plt.title('Effect of Learning Rate on Validation Accuracy')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Validation Accuracy')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / 'accuracy_vs_lr.png', dpi=150)
    plt.close()
    
    # Effect of Kernel Count on Accuracy (group by lr & epochs)
    plt.figure(figsize=(8, 5))
    for lr in sorted(df['learning_rate'].unique()):
        for ep in sorted(df['epochs'].unique()):
            subset = df[(df['learning_rate'] == lr) & (df['epochs'] == ep)]
            plt.plot(subset['kernel_count'], subset['val_accuracy'], 
                     marker='o', label=f'LR={lr}, E={ep}')
    
    plt.title('Effect of Kernel Count on Validation Accuracy')
    plt.xlabel('Kernel Count')
    plt.ylabel('Validation Accuracy')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / 'accuracy_vs_kernels.png', dpi=150)
    plt.close()
    
    # Effect of Epochs on Accuracy (group by lr & kernels)
    plt.figure(figsize=(8, 5))
    for lr in sorted(df['learning_rate'].unique()):
        for kc in sorted(df['kernel_count'].unique()):
            subset = df[(df['learning_rate'] == lr) & 
                       (df['kernel_count'] == kc)]
            plt.plot(subset['epochs'], subset['val_accuracy'], 
                     marker='o', label=f'LR={lr}, K={kc}')
    
    plt.title('Effect of Epochs on Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / 'accuracy_vs_epochs.png', dpi=150)
    plt.close()

# -----------------------
# CLI
# -----------------------

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="CNN Hyperparameter Tuning on EMNIST byclass")
    
    p.add_argument("--full-grid", action="store_true",
                   help="Run the full sweep (lr=[1e-4,1e-3,1e-2,1e-1], "
                        "kernels=[16,32,64,128], epochs=[2,5,10]). "
                        "Default quick grid: lr=[1e-3,1e-2], "
                        "kernels=[32,64], epochs=[2,5]")
    
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0,
                   help="EarlyStopping patience (0 disables)")
    p.add_argument("--limit-train", type=int, default=0, 
                   help="Limit training examples (for smoke tests)")
    p.add_argument("--limit-test", type=int, default=0, 
                   help="Limit test examples (for smoke tests)")
    
    return p.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    set_all_seeds(args.seed)
    
    # Grids
    if args.full_grid:
        learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        kernel_counts = [16, 32, 64, 128]
        epochs_list = [2, 5, 10]
    else:
        learning_rates = [1e-3, 1e-2]
        kernel_counts = [32, 64]
        epochs_list = [2, 5]
    
    # IO
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "results.csv"
    
    # Data
    print("Loading EMNIST/byclass...")
    train_ds, test_ds = build_datasets(
        batch_size=args.batch_size,
        limit_train=args.limit_train,
        limit_test=args.limit_test
    )
    
    # Run
    print("Running grid search...")
    df = run_grid(
        train_ds=train_ds,
        test_ds=test_ds,
        learning_rates=learning_rates,
        kernel_counts=kernel_counts,
        epochs_list=epochs_list,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    
    # Save & report
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results → {csv_path}")
    
    best = df.iloc[0]
    print(f"Best config: lr={best.learning_rate}, "
          f"kernels={int(best.kernel_count)}, "
          f"epochs={int(best.epochs)} | "
          f"val_acc={best.val_accuracy:.4f}")
    
    # Plots
    print("Generating plots...")
    make_plots(df, results_dir)
    print(f"Saved plots to {results_dir}")

if __name__ == "__main__":
    main()
