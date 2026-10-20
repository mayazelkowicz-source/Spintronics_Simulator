#!/usr/bin/env python3
"""
Binarized MLP (BNN) on MNIST for an MTJ-crossbar in-memory-compute simulator.

Architecture : 784 -> 256 -> 10, nn.Linear only, NO bias, NO BatchNorm, NO CNN.
Binarization : weights, inputs and hidden activations are strictly {+1, -1}.
Training     : Straight-Through Estimator (STE) on latent float weights.
Export       : binarized weight matrices -> CSV (crossbar layout), plus a
               numpy integer-only "golden model" check and test vectors.

Run:  python train_bnn_mnist.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
HIDDEN = 256          # hidden layer width (crossbar 2 is HIDDEN x 10)
EPOCHS = 20
BATCH = 128
LR = 2e-3
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# 1) Straight-Through Estimator
# ----------------------------------------------------------------------------
class BinarizeSTE(torch.autograd.Function):
    """
    forward : y = +1 if x >= 0 else -1   (strictly +/-1; sign(0) -> +1, never 0)
    backward: dL/dx = dL/dy * 1_{|x| <= 1}
              i.e. the non-differentiable sign() is bypassed (identity gradient),
              with the standard "hard-tanh" clip that zeroes the gradient where
              the input is saturated (|x| > 1). For weights, this passes the
              gradient straight to the latent continuous weights.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)


binarize = BinarizeSTE.apply


# ----------------------------------------------------------------------------
# 2) Binary linear layer (maps 1:1 onto a crossbar)
# ----------------------------------------------------------------------------
class BinaryLinear(nn.Linear):
    """
    nn.Linear whose weights are binarized on every forward pass.
    The latent float weights live in self.weight (shape [out, in]); only their
    sign is used for compute. No bias -> pure MAC, exactly what a crossbar does.

    Output is divided by sqrt(in_features). This is a POSITIVE constant, so it
    never changes sign(), and the final argmax is unchanged -> hardware does
    not need it. It simply keeps pre-activations ~N(0,1) so the STE clip window
    (|x|<=1) passes useful gradients (this replaces BatchNorm's role).
    """

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        nn.init.uniform_(self.weight, -1.0, 1.0)  # latent weights

    def binary_weight(self):
        return binarize(self.weight)

    def forward(self, x):
        return F.linear(x, self.binary_weight()) / math.sqrt(self.in_features)


# ----------------------------------------------------------------------------
# 3) Model
# ----------------------------------------------------------------------------
class BNN_MLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.fc1 = BinaryLinear(784, hidden)
        self.fc2 = BinaryLinear(hidden, 10)

    @staticmethod
    def binarize_input(x):
        # Pixels in [0,1] -> strictly {-1,+1} (threshold 0.5). The input
        # vector is what you would apply as wordline voltages.
        return torch.where(x.view(x.size(0), -1) > 0.5,
                           torch.ones(1, device=x.device),
                           -torch.ones(1, device=x.device))

    def forward(self, x):
        x = self.binarize_input(x)       # {-1,+1}^784 (no gradient needed)
        x = binarize(self.fc1(x))        # {-1,+1}^HIDDEN, STE backward
        return self.fc2(x)               # integer MAC outputs (scaled) = logits


# ----------------------------------------------------------------------------
# 4) Train / test
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, opt):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        opt.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        opt.step()
        # Keep latent weights in [-1,1]: standard BNN practice; prevents them
        # drifting so far that sign flips become impossible (STE clip window).
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, BinaryLinear):
                    m.weight.clamp_(-1.0, 1.0)
        loss_sum += loss.item() * data.size(0)
        correct += (out.argmax(1) == target).sum().item()
        total += data.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        correct += (model(data).argmax(1) == target).sum().item()
        total += data.size(0)
    return 100.0 * correct / total


# ----------------------------------------------------------------------------
# 5) Hardware weight extraction
# ----------------------------------------------------------------------------
def export_crossbar(model):
    """
    Save binarized weights as CSV in CROSSBAR layout:
        rows    = input lines  (wordlines, one per input neuron)
        columns = output lines (bitlines,  one per output neuron)
    i.e. W has shape [in_features, out_features] and  y = x @ W.
    (PyTorch stores [out, in], so we transpose.)

    Two encodings per layer:
        layerN_weights.csv      : values in {+1,-1}
        layerN_weights_bin.csv  : 0/1 encoding (+1 -> 1, -1 -> 0), e.g. for
                                  MTJ state mapping P/AP (choose your own map).
    """
    mats = []
    for i, layer in enumerate([model.fc1, model.fc2], start=1):
        w = layer.binary_weight().detach().cpu().numpy().T.astype(np.int8)
        assert set(np.unique(w)) <= {-1, 1}, "weights must be strictly +/-1"
        np.savetxt(f"layer{i}_weights.csv", w, fmt="%d", delimiter=",")
        np.savetxt(f"layer{i}_weights_bin.csv", (w > 0).astype(np.int8),
                   fmt="%d", delimiter=",")
        print(f"Saved layer{i}_weights.csv  shape (rows x cols) = {w.shape}")
        mats.append(w)
    return mats


def numpy_golden_inference(x_pm1, W1, W2):
    """
    Integer-only reference of what the crossbars compute (no PyTorch, no
    scaling). x_pm1: [N,784] in {-1,+1}. Use this to validate your C++ sim.
    """
    h = x_pm1.astype(np.int32) @ W1.astype(np.int32)      # crossbar 1 column sums
    h = np.where(h >= 0, 1, -1)                           # sign comparator (0 -> +1)
    y = h @ W2.astype(np.int32)                           # crossbar 2 column sums
    return y                                              # argmax -> class


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    tfm = transforms.ToTensor()  # pixels in [0,1]; binarized inside the model
    train_set = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model = BNN_MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        loss, tr_acc = train_one_epoch(model, train_loader, opt)
        te_acc = evaluate(model, test_loader)
        sched.step()
        print(f"Epoch {epoch:2d}/{EPOCHS} | loss {loss:.4f} | "
              f"train acc {tr_acc:.2f}% | test acc {te_acc:.2f}%")

    final_acc = evaluate(model, test_loader)
    print(f"\nFINAL TEST ACCURACY (PyTorch, binarized): {final_acc:.2f}%")

    # ---- export weights ----
    W1, W2 = export_crossbar(model)

    # ---- verify exported CSVs reproduce accuracy with pure integer math ----
    W1_csv = np.loadtxt("layer1_weights.csv", delimiter=",", dtype=np.int32)
    W2_csv = np.loadtxt("layer2_weights.csv", delimiter=",", dtype=np.int32)
    X = (test_set.data.numpy().reshape(-1, 784) / 255.0 > 0.5)
    X_pm1 = np.where(X, 1, -1).astype(np.int8)
    y = numpy_golden_inference(X_pm1, W1_csv, W2_csv)
    np_acc = 100.0 * (y.argmax(1) == test_set.targets.numpy()).mean()
    print(f"Golden integer-model accuracy from CSVs : {np_acc:.2f}%")

    # ---- test vectors for the C++ simulator (first 100 test images) ----
    np.savetxt("test_inputs_pm1.csv", X_pm1[:100], fmt="%d", delimiter=",")
    np.savetxt("test_expected_logits.csv", y[:100], fmt="%d", delimiter=",")
    np.savetxt("test_labels.csv", test_set.targets.numpy()[:100], fmt="%d")
    print("Saved test_inputs_pm1.csv, test_expected_logits.csv, test_labels.csv")


if __name__ == "__main__":
    main()