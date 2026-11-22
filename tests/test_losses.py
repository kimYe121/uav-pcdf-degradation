"""CMD / L_irr / L_dec 的公式级测试。"""

from __future__ import annotations

import torch

from pcdf.losses import cmd_loss, decoupling_terms, irr_loss


def test_cmd_identical_is_zero() -> None:
    x = torch.randn(8, 256)
    assert float(cmd_loss(x, x)) < 1e-6


def test_cmd_positive_for_shifted() -> None:
    x = torch.randn(16, 64)
    y = x + 1.0
    assert float(cmd_loss(x, y)) > 0.0


def test_irr_zero_for_zero_features() -> None:
    z = torch.zeros(2, 8, 4, 4)
    assert float(irr_loss([z], [z], [z])) == 0.0


def test_irr_shape_mismatch() -> None:
    a = torch.randn(1, 8, 4, 4)
    b = torch.randn(1, 8, 2, 2)
    try:
        irr_loss([a], [a], [b])
    except ValueError:
        return
    raise AssertionError("mismatched shapes must be rejected")


def test_decoupling_total() -> None:
    b, c, h, w = 2, 16, 8, 6
    fs_gap = torch.randn(b, c)
    ft = torch.randn(b, 512)
    adapter = torch.nn.Sequential(torch.nn.Linear(512, c), torch.nn.GELU(), torch.nn.Linear(c, c))
    fv_rgb = [torch.randn(b, c, h, w)]
    fv_ir = [torch.randn(b, c, h, w)]
    fs = [torch.randn(b, c, h, w)]
    l_dc = torch.tensor(3.0)
    out = decoupling_terms(fs_gap, ft, adapter, fv_rgb, fv_ir, fs, 0.01, 0.003, 0.01, l_dc)
    expected = 0.01 * out["l_dt"] + 0.003 * out["l_irr"] + 0.01 * 3.0
    assert torch.allclose(out["l_dec"], expected)
    assert torch.isfinite(out["l_dec"])
