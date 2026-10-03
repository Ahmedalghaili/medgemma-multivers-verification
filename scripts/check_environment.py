#!/usr/bin/env python3
"""Check the runtime before downloading models or launching training."""

import importlib
import platform
import sys


def main():
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    for name in ("torch", "transformers", "pytorch_lightning", "datasets"):
        try:
            module = importlib.import_module(name)
            print(name + ":", getattr(module, "__version__", "installed"))
        except Exception as exc:
            print(name + ": UNAVAILABLE -", repr(exc))
    try:
        import torch

        print("torch.cuda.available:", torch.cuda.is_available())
        print("torch.cuda.count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"cuda[{i}]:", torch.cuda.get_device_name(i))
    except Exception:
        pass


if __name__ == "__main__":
    main()
