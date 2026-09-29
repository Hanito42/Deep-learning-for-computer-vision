"""Student starter code for Lab 04: linear scores and multiclass SVM loss."""

from __future__ import annotations

from typing import Tuple

import torch


def linear_scores(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return class scores for a batch of examples.

    Args:
        X: Tensor of shape ``(N, D)``.
        W: Tensor of shape ``(D, C)``.
        b: Optional bias tensor of shape ``(C,)``.

    Returns:
        Tensor of shape ``(N, C)``.
    """
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must both be rank-2 tensors")
    if X.shape[1] != W.shape[0]:
        raise ValueError("X.shape[1] must equal W.shape[0]")
    if b is not None and b.shape != (W.shape[1],):
        raise ValueError("b must have shape (number_of_classes,)")

    scores = X @ W
    if b is not None:
        scores = scores + b
    return scores


def svm_loss_naive(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute multiclass SVM loss and gradient using explicit loops.

    Lab 04 uses the unregularized objective

    ``mean(data_loss)``.

    The ``reg`` argument remains in the interface so this function can be
    extended in Lab 05. Keep it at ``0.0`` in this lab.

    Args:
        W: Weights of shape ``(D, C)``.
        X: Minibatch of shape ``(N, D)``.
        y: Integer labels of shape ``(N,)``.
        reg: Reserved for Lab 05; must be ``0.0`` in Lab 04.
        delta: SVM margin, normally ``1.0``.

    Returns:
        A scalar loss and a gradient tensor with the same shape as ``W``.
    """
    if X.ndim != 2 or W.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected X=(N,D), W=(D,C), and y=(N,)")
    if X.shape[0] != y.shape[0] or X.shape[1] != W.shape[0]:
        raise ValueError("Input shapes are incompatible")
    if reg != 0.0:
        raise ValueError("Lab 04 uses reg=0.0; L2 regularization begins in Lab 05")

    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = W.new_tensor(0.0)
    dW = torch.zeros_like(W)
    # print(loss)
    # print(dW)
    for i in range(num_train):
        scores = linear_scores(X[i : i + 1], W).squeeze(0)
        # print(i)
        # print(scores)
        # print(X[i : i + 1])
        # print(scores)
        correct_score = scores[y[i]]
        # print(correct_score)
        print(y)
        for j in range(num_classes):
            # print(j)
            # print(y[i])
            if j == y[i]:
                # print('skip')
                continue
            margin = scores[j] - correct_score + delta
            if margin > 0:
                # TODO 1: accumulate this positive margin into ``loss``.
                loss+=margin
                # TODO 2: add this example's contribution to ``dW``.
                dW[:, j] += X[i]
                dW[:, y[i]] -=X[i]
    # TODO 3: average the data loss and data gradient over the minibatch.
    # print(dW)
    loss = loss / num_train
    dW = dW / num_train

    return loss, dW
    raise NotImplementedError("Complete svm_loss_naive for Lab 04")


from pathlib import Path
import torch

import dlcv2026
# Save linear_classifier.py, then rerun this cell.
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
W = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
y = torch.tensor([2, 0, 1], dtype=torch.int64)

loss, gradient = svm_loss_naive(
    W.clone(), X, y, reg=0.0
)
print("unregularized loss:", float(loss))
print("gradient:\n", gradient)

assert loss.ndim == 0
assert gradient.shape == W.shape
assert torch.allclose(loss, torch.tensor(2.0), atol=1e-12)
