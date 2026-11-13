"""Environment sanity check for the conda base interpreter."""

from __future__ import annotations

import sys


def main() -> None:
    import cv2
    import numpy as np
    import torch
    import ultralytics

    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    print(f"cuda_ok      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device       {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram         {free / 1024**3:.1f} GB free / {total / 1024**3:.1f} GB total")
    print(f"ultralytics  {ultralytics.__version__}")
    print(f"opencv       {cv2.__version__}")
    print(f"numpy        {np.__version__}")

    from ultralytics import YOLO

    model = YOLO("yolo11n-obb.yaml")
    params = sum(p.numel() for p in model.model.parameters())
    print(f"yolo11n-obb  built OK, {params / 1e6:.2f}M params")


if __name__ == "__main__":
    main()
