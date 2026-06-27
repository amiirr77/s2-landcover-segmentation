"""Offline sanity check — no network, no real data.

Verifies the model builds and one optimisation step runs end to end. Run with:
    python -m tests.test_smoke
or under pytest.
"""

import numpy as np
import torch

from s2seg import NUM_CLASSES
from s2seg.model import build_model


def test_forward_backward_step():
    torch.manual_seed(0)
    # encoder_weights=None so it works fully offline
    model = build_model(classes=NUM_CLASSES, in_channels=4, encoder_weights=None)
    x = torch.from_numpy(np.random.rand(2, 4, 256, 256).astype("float32"))
    y = torch.from_numpy(np.random.randint(0, NUM_CLASSES, (2, 256, 256)).astype("int64"))

    logits = model(x)
    assert logits.shape == (2, NUM_CLASSES, 256, 256), logits.shape

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite"
    print(f"OK — logits {tuple(logits.shape)}, loss {loss.item():.4f}")


if __name__ == "__main__":
    test_forward_backward_step()
