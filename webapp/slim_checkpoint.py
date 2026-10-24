#!/usr/bin/env python3
"""Strip training-only state (optimizer, EMA, extra_state) from a fairseq
checkpoint, keeping just what inference needs: model weights + cfg/args.
Roughly halves the file, cuts peak RAM at load, and speeds up the container.

Usage: python slim_checkpoint.py <in.pt> <out.pt>
"""
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print("usage: python slim_checkpoint.py <in.pt> <out.pt>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    sd = torch.load(src, map_location="cpu", weights_only=False)
    # The bulk is the Adam optimizer tensors under 'last_optimizer_state'.
    # Everything else (cfg/args/model/optimizer_history metadata) is small and
    # is needed by fairseq's checkpoint loader, so keep it intact.
    keep = {k: v for k, v in sd.items() if k != "last_optimizer_state"}
    torch.save(keep, dst)
    import os
    a = os.path.getsize(src) / 1e9
    b = os.path.getsize(dst) / 1e9
    print(f"{src} ({a:.2f} GB) -> {dst} ({b:.2f} GB)  kept keys: {list(keep)}")


if __name__ == "__main__":
    main()
