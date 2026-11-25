"""输入几何回归测试：源图必须原尺寸进入网络。

``RandomPerspective.size`` 是 **(W, H)**，而本仓库的 ``DataConfig.imgsz`` 是
**(H, W)**。两者传反时的表现很隐蔽：图先被 warp 成竖图，再由 ``LetterBox``
等比缩到 0.8 并补灰边——形状仍是 ``imgsz``，只有灰边和 content 缩放能暴露。

本测试用合成图锁住两条约束：

* 正确顺序：640×512 源图 → 512×640 张量，**没有任何 padding**（ratio=1）；
* 传反顺序：同样输出形状，但出现 padding（= 内容被缩小）。
"""

from __future__ import annotations

import numpy as np
from ultralytics.data.augment import Compose, LetterBox, RandomPerspective
from ultralytics.utils.instance import Instances

IMGSZ = (512, 640)  # (H, W)
PAD = 114


def _run(size_w_h: tuple[int, int]) -> np.ndarray:
    height, width = IMGSZ
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    transform = Compose(
        [
            RandomPerspective(
                degrees=0.0, translate=0.0, scale=(1.0, 1.0), shear=0.0,
                perspective=0.0, size=size_w_h, preserve_obb=True,
            ),
            LetterBox(
                new_shape=(height, width), auto=False, scaleup=False,
                stride=32, padding_value=PAD,
            ),
        ]
    )
    empty = Instances(np.zeros((0, 4), np.float32))
    labels = {"img": img, "instances": empty, "cls": np.zeros((0, 1), np.float32)}
    return transform(labels)["img"]


def test_native_geometry_has_no_padding() -> None:
    height, width = IMGSZ
    out = _run((width, height))  # Ultralytics 要 (W, H)
    assert out.shape == (height, width, 3)
    assert not (out == PAD).any(), "原尺寸输入不应出现灰边（灰边意味着被等比缩小了）"


def test_transposed_size_downscales_and_pads() -> None:
    height, width = IMGSZ
    out = _run((height, width))  # 传反：warp 成竖图再被 LetterBox 压回
    assert out.shape == (height, width, 3)
    assert (out == PAD).any(), "传反 (H,W) 时必须能观察到灰边，否则本测试失去意义"