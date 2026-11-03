#!/usr/bin/env python3
"""
Export the full 10,000-image MNIST test set for the C++ simulator.

Uses the existing layer1_weights.csv / layer2_weights.csv as they are (no
training), reproduces the integer golden model of main.py, checks that the
first 100 samples match the committed test vectors, and writes:

    full_test/test_inputs_pm1.csv
    full_test/test_expected_logits.csv
    full_test/test_labels.csv

Run from the repository root:  python3 export_full_test.py
"""

import os
import sys

import numpy as np
from torchvision import datasets

OUT_DIR = "full_test"
NEWLINE = "\r\n"  # the committed test vectors use CRLF line endings


def numpy_golden_inference(x_pm1, W1, W2):
    """Identical to numpy_golden_inference in main.py."""
    h = x_pm1.astype(np.int32) @ W1.astype(np.int32)      # crossbar 1 column sums
    h = np.where(h >= 0, 1, -1)                           # sign comparator (0 -> +1)
    y = h @ W2.astype(np.int32)                           # crossbar 2 column sums
    return y                                              # argmax -> class


def load_existing(path):
    return np.loadtxt(path, delimiter=",", dtype=np.int32, ndmin=2)


def check_prefix(name, actual, path):
    expected = load_existing(path)
    n = expected.shape[0]
    actual = np.asarray(actual[:n], dtype=np.int32).reshape(n, -1)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        bad = [i for i in range(min(len(expected), len(actual)))
               if not np.array_equal(expected[i], actual[i])]
        sys.exit(f"ERROR: first-100 check failed for {name} against {path}: "
                 f"shape {actual.shape} vs {expected.shape}, "
                 f"mismatching rows (first 10): {bad[:10]}")
    print(f"[OK] first {len(expected)} {name} match {path}")


def main():
    W1 = np.loadtxt("layer1_weights.csv", delimiter=",", dtype=np.int32)
    W2 = np.loadtxt("layer2_weights.csv", delimiter=",", dtype=np.int32)
    assert W1.shape == (784, 256), f"layer1_weights.csv shape {W1.shape}, expected (784, 256)"
    assert W2.shape == (256, 10), f"layer2_weights.csv shape {W2.shape}, expected (256, 10)"

    test_set = datasets.MNIST("./data", train=False, download=True)
    X = (test_set.data.numpy().reshape(-1, 784) / 255.0 > 0.5)
    X_pm1 = np.where(X, 1, -1).astype(np.int8)
    labels = test_set.targets.numpy()

    y = numpy_golden_inference(X_pm1, W1, W2)
    acc = 100.0 * (y.argmax(1) == labels).mean()
    print(f"Golden integer-model accuracy on {len(labels)} test images: {acc:.2f}%")

    check_prefix("inputs", X_pm1, "test_inputs_pm1.csv")
    check_prefix("logits", y, "test_expected_logits.csv")
    check_prefix("labels", labels, "test_labels.csv")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savetxt(os.path.join(OUT_DIR, "test_inputs_pm1.csv"), X_pm1, fmt="%d", delimiter=",",
               newline=NEWLINE)
    np.savetxt(os.path.join(OUT_DIR, "test_expected_logits.csv"), y, fmt="%d", delimiter=",",
               newline=NEWLINE)
    np.savetxt(os.path.join(OUT_DIR, "test_labels.csv"), labels, fmt="%d", newline=NEWLINE)
    print(f"Saved {len(labels)} samples to {OUT_DIR}/")


if __name__ == "__main__":
    main()
