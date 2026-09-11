"""MS-TCN++ head: shapes, the causal-padding guarantee, and that a few steps reduce the loss."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sop_monitor.mstcn import MSTCNSpec, _loss, build_model  # noqa: E402

SMALL = MSTCNSpec(
    num_layers_pg=4, num_layers_r=3, num_refinement_stages=2, num_f_maps=8, dropout=0.0
)


def test_outputs_one_logit_map_per_stage() -> None:
    model = build_model(dim=6, n_classes=5, spec=SMALL, causal=True).eval()
    outputs = model(torch.randn(1, 6, 40))
    assert len(outputs) == 1 + SMALL.num_refinement_stages
    assert all(tuple(o.shape) == (1, 5, 40) for o in outputs)


def test_causal_model_ignores_the_future_and_offline_model_does_not() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 6, 60)
    altered = x.clone()
    altered[:, :, 30:] = 0.0
    causal = build_model(6, 5, SMALL, causal=True).eval()
    with torch.no_grad():
        before, after = causal(x)[-1], causal(altered)[-1]
    assert torch.allclose(before[:, :, :30], after[:, :, :30], atol=1e-5)
    assert not torch.allclose(before[:, :, 30:], after[:, :, 30:], atol=1e-5)
    offline = build_model(6, 5, SMALL, causal=False).eval()
    with torch.no_grad():
        before, after = offline(x)[-1], offline(altered)[-1]
    assert not torch.allclose(before[:, :, :30], after[:, :, :30], atol=1e-5)


def test_a_few_optimiser_steps_reduce_the_loss_on_a_toy_sequence() -> None:
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    labels = np.repeat(rng.integers(0, 3, size=6), 10)
    x = torch.from_numpy(
        (np.eye(3)[labels].T + 0.1 * rng.standard_normal((3, 60))).astype(np.float32)
    ).unsqueeze(0)
    y = torch.from_numpy(labels).unsqueeze(0)
    model = build_model(3, 3, SMALL, causal=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        optimiser.zero_grad()
        loss = _loss(model(x), y, 3, SMALL)
        loss.backward()
        optimiser.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first * 0.5
